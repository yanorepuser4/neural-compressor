# -*- coding: utf-8 -*-
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Common constants for easier e2e tests creation."""
from enum import Enum

DEFAULT_INPUT_MODEL_NAME = "Input model"


class BenchmarkMode(Enum):
    """Define benchmark mode selection options."""

    ACCURACY = "accuracy"
    PERFORMANCE = "performance"


class ButtonCSSSelector(Enum):
    """Define CSS selectors for various buttons."""

    ADD_NEW_OPTIMIZATION = (
        "button.mat-focus-indicator.create-new-btn.mat-raised-button.mat-button-base:not"
        "(#create-new-project-menu-btn):not( #system-info-btn)"
    )
    ADD_NEW_BENCHMARK = (
        "button.mat-focus-indicator.create-new-btn.mat-raised-button.mat-button-base"
        ":not(#create-new-project-menu-btn)"
    )
    NEXT = 'div[style*="transform: none"] button[name="next"]'
    COMPARE_SELECTED = "button.compare-btn"


class StatusCSSSelector(Enum):
    """Define CSS selectors for job status icon."""

    WIP = ".mat-tooltip-trigger > svg > .ng-star-inserted"
    SUCCESS = "img[src='./../../assets/010a-passed-completed-solid.svg']"


class TabCSSSelector(Enum):
    """Define CSS selectors for selecting tab (Optimizations, Benchmarks, ...)."""

    BENCHMARKS = 'div[joyridestep="benchmarkTour"]'


class DomainCSSSelector(Enum):
    """Define CSS selectors for domain selecting during predefined project creation."""

    IMAGE_RECOGNITION = "#domain0"
    OBJECT_DETECTION = "#domain1"


class ModelCSSSelector(Enum):
    """Define CSS selectors for model during predefined model project creation."""

    INCEPTION_V3 = "#model0"


class PrecisionCSSSelector(Enum):
    """Define CSS selectors for precision durin adding optimization."""

    pass
