# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def get_demo_model():
    return FakeModel()


def get_dummy_dataloader():
    return FakeCalibDataloder()


def get_default_quant_config():
    return FakeQuantConfig()


def get_default_tuning_criterion():
    return FakeTuningConfig()


def get_default_quant_with_tuning_criterion():
    return FakeQuantConfig()


FAKE_EVAL_RESULT = 1.0


class FakeQuantConfig:
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "FakeQuantConfig"


class FakeTuningConfig:
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "FakeTuningConfig"


class FakeCalibDataloder:
    def __init__(self):
        pass

    def __repr__(self):
        return "FakeCalibDataloder"


class FakeModel:
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "FakeModel"
