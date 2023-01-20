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
"""Class testing onnx resnet50 model."""

import os
import time
from test.ux.selenium.constants import (
    DEFAULT_INPUT_MODEL_NAME,
    BenchmarkMode,
    ButtonCSSSelector,
    StatusCSSSelector,
    TabCSSSelector,
)
from test.ux.selenium.e2e_test_base import E2ETestBase
from test.ux.selenium.utils import (
    add_benchmark,
    add_optimization,
    check_show_on_chart,
    create_project,
    start_benchmark,
    start_optimization,
)

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait


class TestONNXResnet50(E2ETestBase):
    """Base class for end to end tests."""

    def test_optimization_and_benchmarks(self, params):
        """
        Test following scenario.

        Add new project. Add optimization in this project. Run optimization.
        Wait for it to end. Add benchmark with optimized model and with input model.
        Run both benchmarks. Wait for benchmarks to end. Add both benchmarks to compare.
        Show compare graph.
        """
        address = params["address"]
        port = params["port"]
        url_prefix = params["url_prefix"]
        token = params["token"]
        inc_url = f"https://{address}:{port}/{url_prefix}/home?token={token}"
        models_dir = params["models_dir"]
        model_path = os.path.join(models_dir, "resnet50-v1-12.onnx")
        self.driver.get(inc_url)

        optimized_model_name = "default_optimization"
        project_name = "Test_project"
        create_project(
            self.driver,
            project_name,
            model_path=model_path,
        )
        add_optimization(self.driver, optimized_model_name)
        start_optimization(self.driver, optimized_model_name)
        # Wait for Success icon to appear; optimization end.
        try:
            WebDriverWait(self.driver, 200).until(
                expected_conditions.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        StatusCSSSelector.SUCCESS.value,
                    ),
                ),
            )
        except TimeoutException:
            self.driver.save_screenshot("waiting_for_opt_failed.png")
            raise

        # Select Benchmarks tab
        self.driver.find_element(By.CSS_SELECTOR, TabCSSSelector.BENCHMARKS.value).click()
        # Wait for tab to change to Benchmarks
        WebDriverWait(self.driver, 10).until(
            expected_conditions.visibility_of_element_located(
                (
                    By.CSS_SELECTOR,
                    ButtonCSSSelector.COMPARE_SELECTED.value,
                ),
            ),
        )
        opt_benchmark_name = "perf_optimized"
        in_benchmark_name = "perf_input"
        add_benchmark(
            self.driver,
            opt_benchmark_name,
            optimized_model_name,
            BenchmarkMode.PERFORMANCE,
        )
        add_benchmark(
            self.driver,
            in_benchmark_name,
            DEFAULT_INPUT_MODEL_NAME,
            BenchmarkMode.PERFORMANCE,
        )
        time.sleep(5)
        start_benchmark(self.driver, opt_benchmark_name)
        try:
            start_benchmark(self.driver, in_benchmark_name)
        except ValueError:
            self.driver.save_screenshot("in_benchmark_start_fail.png")
            raise
        # Wait for any benchmark to start
        WebDriverWait(self.driver, 30).until(
            expected_conditions.visibility_of_element_located(
                (
                    By.CSS_SELECTOR,
                    StatusCSSSelector.WIP.value,
                ),
            ),
        )
        try:
            # Wait for last benchmark to finish; all wip status icons must change to other status.
            WebDriverWait(self.driver, 100).until(
                expected_conditions.invisibility_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        StatusCSSSelector.WIP.value,
                    ),
                ),
            )
        except TimeoutException:
            self.driver.save_screenshot("waiting_benchmark_end_failed.png")
            raise

        # Crete comparison chart.
        check_show_on_chart(self.driver, opt_benchmark_name)
        check_show_on_chart(self.driver, in_benchmark_name)
        self.driver.find_element(
            By.CSS_SELECTOR,
            ButtonCSSSelector.COMPARE_SELECTED.value,
        ).click()
        WebDriverWait(self.driver, 10).until(
            expected_conditions.visibility_of_element_located(
                (
                    By.CSS_SELECTOR,
                    "ngx-charts-bar-vertical-2d svg",
                ),
            ),
        )
        self.driver.save_screenshot("comp_chart.png")
