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
"""Utilities for selenium tests."""

import time
from test.ux.selenium.constants import (
    BenchmarkMode,
    ButtonCSSSelector,
    DomainCSSSelector,
    ModelCSSSelector,
    PrecisionCSSSelector,
)
from typing import Optional

from selenium.common.exceptions import ElementClickInterceptedException
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait


def get_number_of_projects(driver):
    """Get current number of projects."""
    project_list = driver.find_element(By.CLASS_NAME, "project-list")
    return len(project_list.find_elements(By.CSS_SELECTOR, "span.name"))


"""
Every function needs their Prerequisites to be met, to run successfully.
"""


def create_project(
    driver: Chrome,
    project_name,
    model_path: Optional[str] = None,
    domain: Optional[DomainCSSSelector] = None,
    model: Optional[ModelCSSSelector] = None,
):
    """
    Create new project.

    Prerequisites:
    1. Navigational panel on the left side of screen unfolded.
    2. Create new project button on the left panel visible;
    might be hidden if there is many project on the list.
    End location:
        Freshly created project Optimizations tab.
    """
    driver.find_element(By.ID, "create-new-project-menu-btn").click()
    project_name_input = WebDriverWait(driver, 5).until(
        expected_conditions.element_to_be_clickable(
            (By.ID, "project_name"),
        ),
    )
    # Clear input text
    project_name_input.clear()
    # Insert project name
    project_name_input.send_keys(project_name)
    time.sleep(1)
    driver.find_element(By.CSS_SELECTOR, ButtonCSSSelector.NEXT.value).click()
    # Custom model
    if model_path and domain is None and model is None:
        # Select custom model radio button
        WebDriverWait(driver, 1).until(
            expected_conditions.element_to_be_clickable(
                (By.CSS_SELECTOR, "label[for=custom-radio-input]"),
            ),
        ).click()
        time.sleep(1)
        driver.find_element(By.CSS_SELECTOR, ButtonCSSSelector.NEXT.value).click()
        WebDriverWait(driver, 1).until(
            expected_conditions.element_to_be_clickable(
                (By.ID, "model_path"),
            ),
        ).send_keys(model_path)
    # Predefined model
    elif domain and model and model_path is None:
        raise NotImplementedError("Create predefined models not implemented.")
    else:
        ValueError(
            "Specify model_path to create custom project"
            "or domain and model to create predefined model.",
        )

    # Click finish button
    finish_btn = WebDriverWait(driver, 10).until(
        expected_conditions.element_to_be_clickable(
            (By.ID, "finish-adv-btn"),
        ),
    )
    finish_btn.click()
    # Wait for project tab to load
    WebDriverWait(driver, 10).until(
        expected_conditions.text_to_be_present_in_element(
            (By.CSS_SELECTOR, 'h1[joyridestep="intro"]'),
            project_name,
        ),
    )


def add_optimization(driver, result_model_name, precision: Optional[PrecisionCSSSelector] = None):
    """
    Create optimization with Optimization name == result_model_name.

    Prerequisites:
    1. Add new optimization button clickable.
    """
    add_new_opt_btn = driver.find_element(
        By.CSS_SELECTOR,
        ButtonCSSSelector.ADD_NEW_OPTIMIZATION.value,
    )
    # Click add new optimization button
    add_new_opt_btn.click()
    # Clear input text
    name_input = driver.find_element(By.CSS_SELECTOR, 'input[formcontrolname="name"]')
    name_input.clear()
    name_input.send_keys(result_model_name)
    time.sleep(1)
    driver.find_element(By.CSS_SELECTOR, ButtonCSSSelector.NEXT.value).click()
    if precision:
        raise NotImplementedError("Precision selection not implemented.")

    finish_btn = WebDriverWait(driver, 1).until(
        expected_conditions.element_to_be_clickable(
            (By.ID, "finish-adv-btn"),
        ),
    )
    finish_btn.click()
    WebDriverWait(driver, 3).until(
        expected_conditions.element_to_be_clickable(
            (By.CSS_SELECTOR, "td > .action-btn"),
        ),
    )


def add_benchmark(
    driver: Chrome,
    benchmark_name,
    model_name,
    benchmark_mode: BenchmarkMode,
    dataset_name=None,
):
    """
    Add benchmark when in benchmark tab.

    Prerequisites:
    1. Add new benchmark button clickable.
    """
    # Press add new benchmark
    new_benchmark_btn = driver.find_element(
        By.CSS_SELECTOR,
        ButtonCSSSelector.ADD_NEW_BENCHMARK.value,
    )
    new_benchmark_btn.click()
    # Wait for pop-up window to appear
    benchmark_name_input = WebDriverWait(driver, 50).until(
        expected_conditions.element_to_be_clickable(
            (
                By.CSS_SELECTOR,
                'input[formcontrolname="name"]',
            ),
        ),
    )
    # Clear input text
    benchmark_name_input.clear()
    benchmark_name_input.send_keys(benchmark_name)
    time.sleep(1)
    driver.find_element(By.CSS_SELECTOR, ButtonCSSSelector.NEXT.value).click()

    if benchmark_mode is BenchmarkMode.PERFORMANCE:
        # Performance is selected by default
        ...
    else:
        raise NotImplementedError(f"Passed benchmark_mode {benchmark_mode} not implemented.")
    WebDriverWait(driver, 50).until(
        expected_conditions.element_to_be_clickable(
            (
                By.CSS_SELECTOR,
                ButtonCSSSelector.NEXT.value,
            ),
        ),
    ).click()

    model_combo_box = WebDriverWait(driver, 1).until(
        expected_conditions.element_to_be_clickable(
            (By.CSS_SELECTOR, 'mat-select[role="combobox"][formcontrolname="modelId"]'),
        ),
    )
    # Unfold models list
    model_combo_box.click()
    time.sleep(1)
    # Select optimized model
    models = driver.find_elements(By.CSS_SELECTOR, "div[role='listbox']>mat-option>span")
    # Find index on list of model_name
    model_idx = 1
    for model in models:
        if model.text == model_name:
            break
        model_idx += 1
    if model_idx == len(models) + 1:
        driver.save_screenshot("model_list_add_benchmark_fail.png")
        raise ValueError(f"Could not find model_name={model_name} in models list.")

    driver.find_element(
        By.CSS_SELECTOR,
        f'div[role="listbox"] mat-option[role="option"]:nth-of-type({model_idx})',
    ).click()
    WebDriverWait(driver, 5).until(
        expected_conditions.element_to_be_clickable(
            (
                By.CSS_SELECTOR,
                ButtonCSSSelector.NEXT.value,
            ),
        ),
    ).click()
    if dataset_name:
        raise NotImplementedError()
    else:
        # Leave dataset default (dummy)
        pass
    WebDriverWait(driver, 5).until(
        expected_conditions.element_to_be_clickable(
            (
                By.CSS_SELECTOR,
                ButtonCSSSelector.NEXT.value,
            ),
        ),
    ).click()
    # Uncheck iterate over whole dataset checkbox
    try:
        WebDriverWait(driver, 5).until(
            expected_conditions.element_to_be_clickable(
                (By.CSS_SELECTOR, "mat-checkbox"),
            ),
        ).click()
    except ElementClickInterceptedException:
        pass  # TODO: Find solution so it works when other benchmark already exists.

    WebDriverWait(driver, 4).until(
        expected_conditions.element_to_be_clickable(
            (By.ID, "finish-adv-btn"),
        ),
    ).click()


def start_optimization(
    driver: Chrome,
    optimization_name=None,
):
    """
    Start optimization with name==optimization_name param.

    If optimization_name is None start first optimization on the list.
    Function returns immediately after starting selected optimization.

    Prerequisites:
    1. Optimizations tab.
    2. Optimization with Optimization name == optimization_name exists.
    """
    if optimization_name:
        optimizations = driver.find_elements(
            By.CSS_SELECTOR,
            "app-optimizations>table>tr[mattooltip]>td:first-child",
        )
        idx = 1
        for optimization in optimizations:
            if optimization.text == optimization_name:
                break
            idx += 1
        # Skip optimizations table's header
        idx += 1
        driver.find_element(
            By.CSS_SELECTOR,
            f"app-optimizations>table>tr:nth-of-type({idx})>td button",
        ).click()
    else:
        # Start first optimization on list
        driver.find_element(By.CSS_SELECTOR, "td > .action-btn").click()


def start_benchmark(driver, benchmark_name=None):
    """
    Start benchmark with name==benchmark_name param.

    If optimization_name is None start first benchmark on the list.
    Function returns immediately after starting selected benchmark.

    Prerequisites:
    1. Benchmarks tab.
    2. Benchmark with Name==benchmark_name exists.
    """
    if benchmark_name:
        row_selector = get_selector_of_benchmark_row(driver, benchmark_name)
        run_btn = driver.find_element(
            By.CSS_SELECTOR,
            row_selector + '>td>button[color="accent"]',
        )
    else:
        # Start first possible to run optimization on list
        run_btn = driver.find_element(
            By.CSS_SELECTOR,
            "td > .action-btn",
        )
    run_btn.click()


def get_selector_of_benchmark_row(driver, benchmark_name):
    """
    Return CSS selector of row where name field==banchmark_name.

    Prerequisites:
    1. Benchmarks tab.
    2. Benchmark with Name==benchmark_name must exist.
    """
    benchmarks = driver.find_elements(
        By.CSS_SELECTOR,
        "app-benchmarks>table>tr[mattooltip]>td:first-child",
    )
    idx = 1
    idx += 1  # skip table header
    for benchmark in benchmarks:
        if benchmark.text == benchmark_name:
            break
        idx += 1
    if idx == len(benchmarks) + 2:
        raise ValueError(f"Benchmark with {benchmark_name} does not exist.")
    return f"app-benchmarks>table>tr:nth-of-type({idx})"


def check_show_on_chart(driver, benchmark_name):
    """
    Check show on chart checkbox in benchmark tab.

    Prerequisites:
    1. Benchmarks tab.
    2. Benchmark with benchmark_name=Name must exist.
    """
    row_selector = get_selector_of_benchmark_row(
        driver,
        benchmark_name,
    )
    driver.find_element(
        By.CSS_SELECTOR,
        row_selector + ">td>mat-checkbox",
    ).click()
