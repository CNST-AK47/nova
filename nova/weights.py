# Copyright (c) 2011-2012 OpenStack Foundation
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

"""
Pluggable Weighing support
"""

import abc

from oslo_log import log as logging

from nova import loadables


LOG = logging.getLogger(__name__)


def normalize(weight_list, minval=None, maxval=None):
    """Normalize the values in a list between 0 and 1.0.

    The normalization is made regarding the lower and upper values present in
    weight_list. If the minval and/or maxval parameters are set, these values
    will be used instead of the minimum and maximum from the list.

    If all the values are equal, they are normalized to 0.
    """

    if not weight_list:
        return ()

    if maxval is None:
        maxval = max(weight_list)

    if minval is None:
        minval = min(weight_list)

    maxval = float(maxval)
    minval = float(minval)

    if minval == maxval:
        return [0] * len(weight_list)

    range_ = maxval - minval
    return ((i - minval) / range_ for i in weight_list)


class WeighedObject(object):
    """Object with weight information."""

    def __init__(self, obj, weight):
        self.obj = obj
        self.weight = weight

    def __repr__(self):
        return "<WeighedObject '%s': %s>" % (self.obj, self.weight)


class BaseWeigher(metaclass=abc.ABCMeta):
    """Base class for pluggable weighers.
    基础称重器类
    The attributes maxval and minval can be specified to set up the maximum
    and minimum values for the weighed objects. These values will then be
    taken into account in the normalization step, instead of taking the values
    from the calculated weights.
    """

    minval = None
    maxval = None

    def weight_multiplier(self, host_state):
        """How weighted this weigher should be.

        Override this method in a subclass, so that the returned value is
        read from a configuration option to permit operators specify a
        multiplier for the weigher. If the host is in an aggregate, this
        method of subclass can read the ``weight_multiplier`` from aggregate
        metadata of ``host_state``, and use it to overwrite multiplier
        configuration.

        :param host_state: The HostState object.
        """
        return 1.0

    @abc.abstractmethod
    def _weigh_object(self, obj, weight_properties):
        """Weigh an specific object."""

    def weigh_objects(self, weighed_obj_list, weight_properties):
        """Weigh multiple objects.

        Override in a subclass if you need access to all objects in order
        to calculate weights. Do not modify the weight of an object here,
        just return a list of weights.
        """
        # Calculate the weights
        weights = []
        # 遍历对象列表
        for obj in weighed_obj_list:
            # 计算权重
            weight = self._weigh_object(obj.obj, weight_properties)

            # don't let the weight go beyond the defined max/min
            if self.minval is not None:
                weight = max(weight, self.minval)
            if self.maxval is not None:
                weight = min(weight, self.maxval)

            weights.append(weight)

        return weights


class BaseWeightHandler(loadables.BaseLoader):
    object_class = WeighedObject

    def get_weighed_objects(self, weighers, obj_list, weighing_properties):
        """Return a sorted (descending), normalized list of WeighedObjects."""
        # 进行对象转换
        weighed_objs = [self.object_class(obj, 0.0) for obj in obj_list]

        if len(weighed_objs) <= 1:
            return weighed_objs

        for weigher in weighers:
            # 计算对象权重
            weights = weigher.weigh_objects(weighed_objs, weighing_properties)

            LOG.debug(
                "%s: raw weights %s",
                weigher.__class__.__name__,
                {(obj.obj.host, obj.obj.nodename): weight
                 for obj, weight in zip(weighed_objs, weights)}
            )

            # Normalize the weights
            # 将权重进行归一化--为每个宿主机资源对象设置最大最小值，然后进行归一化
            # 保证每个权重值都在0-1之间
            weights = list(
                normalize(
                    weights, minval=weigher.minval, maxval=weigher.maxval))

            LOG.debug(
                "%s: normalized weights %s",
                weigher.__class__.__name__,
                {(obj.obj.host, obj.obj.nodename): weight
                 for obj, weight in zip(weighed_objs, weights)}
            )

            log_data = {}

            for i, weight in enumerate(weights):
                obj = weighed_objs[i]
                # 获取权重
                multiplier = weigher.weight_multiplier(obj.obj)
                weigher_score = multiplier * weight
                obj.weight += weigher_score

                log_data[(obj.obj.host, obj.obj.nodename)] = (
                    f"{multiplier} * {weight}")

            LOG.debug(
                "%s: score (multiplier * weight) %s",
                weigher.__class__.__name__,
                {name: log for name, log in log_data.items()}
            )
        # 进行权重排序
        return sorted(weighed_objs, key=lambda x: x.weight, reverse=True)
