#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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
# pylint: disable=no-member

"""DyNAS for INC."""

import argparse
import csv
import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable

from neural_compressor.conf.config import NASConfig
from neural_compressor.experimental.nas import NAS as ExpNAS
from neural_compressor.experimental.nas.dynast.dynas_utils import TorchVisionReference


def parse_args() -> Any:
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search_algorithm",
        type=str,
        required=True,
        help="search algorithm.",
    )
    parser.add_argument(
        "--supernet",
        type=str,
        required=True,
        help="supernet.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="seed.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        type=str,
        required=True,
        help="metrics.",
    )
    parser.add_argument(
        "--population",
        type=int,
        required=True,
        help="population.",
    )
    parser.add_argument(
        "--num_evals",
        type=int,
        required=True,
        help="num_evals.",
    )
    parser.add_argument(
        "--results_csv_path",
        type=str,
        required=True,
        help="results_csv_path.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="batch_size.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="dataset_path.",
    )
    parser.add_argument(
        "--work_path",
        type=str,
        required=True,
        help="work_path.",
    )
    parser.add_argument(
        "--img_file_name",
        type=str,
        required=True,
        help="img_file_name.",
    )
    return parser.parse_args()


class DyNAS:
    """DyNAS for INC GUI."""

    def __init__(self, config: dict) -> None:
        """Initialize nas config."""
        self.config = NASConfig(approach="dynas", search_algorithm=config.get("search_algorithm"))
        self.config.dynas.supernet = config.get("supernet")
        self.config.seed = config.get("seed")
        self.config.dynas.metrics = config.get("metrics")
        self.config.dynas.population = config.get("population")
        self.config.dynas.num_evals = config.get("num_evals")
        self.config.dynas.results_csv_path = config.get("results_csv_path")
        self.config.dynas.batch_size = config.get("batch_size")
        self.config.dynas.dataset_path = config.get("dataset_path")

    def __call__(self, img_path: str) -> Any:
        """Run nas."""
        self.perform_search()
        refer_result = self.measure_reference_architecture()
        self.plot_search_results(img_path)
        return refer_result

    def perform_search(self) -> None:
        """Perform search."""
        agent = ExpNAS(self.config)
        agent.num_workers = 0
        self.generate_csv(self.config.dynas.results_csv_path)
        agent.search()

    def generate_csv(self, name: str) -> None:
        """Generate csv."""
        if not os.path.exists(name):
            with open(name, "w", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Sub-network", "Date", "Latency (ms)", "MACs", "Top-1 Acc (%)"])

    def measure_reference_architecture(self) -> dict:
        """Measure reference architecture."""
        ref = TorchVisionReference(
            self.config.dynas.supernet,
            self.config.dynas.dataset_path,
            self.config.dynas.batch_size,
            num_workers=0,
        )
        latency = ref.measure_latency()
        macs = ref.validate_macs()
        loss, top1, top5 = ref.validate_top1()
        return {"latency": latency, "macs": macs, "top1": top1, "loss": loss, "top5": top5}

    def plot_search_results(self, img_path: str) -> str:
        """Plot search results."""
        fig, fig_ax = plt.subplots(figsize=(7, 5))

        number_of_evals = self.config.dynas.num_evals
        supernet_name = self.config.dynas.supernet
        df_dynas = pd.read_csv(self.config.dynas.results_csv_path)[:number_of_evals]
        df_dynas.columns = ["config", "date", "lat", "macs", "top1"]

        cmap = plt.cm.get_cmap("viridis_r")
        count = list(range(len(df_dynas)))

        fig_ax.scatter(
            df_dynas["macs"].values,
            df_dynas["top1"].values,
            marker="^",
            alpha=0.8,
            c=count,
            cmap=cmap,
            label="Discovered DNN Model",
            s=10,
        )
        fig_ax.set_title(
            f"Intel® Neural Compressor\nDynamic NAS (DyNAS)\nSupernet:{supernet_name}",
        )
        fig_ax.set_xlabel("MACs", fontsize=13)
        fig_ax.set_ylabel("Top-1 Accuracy (%)", fontsize=13)
        fig_ax.legend(fancybox=True, fontsize=10, framealpha=1, borderpad=0.2, loc="lower right")
        fig_ax.grid(True, alpha=0.3)

        # Eval Count bar
        norm = plt.Normalize(0, len(df_dynas))
        scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(scalar_mappable, ax=fig_ax, shrink=0.85)
        cbar.ax.set_title("         Evaluation\n  Count", fontsize=8)

        fig.tight_layout(pad=2)
        plt.show()
        plt.savefig(img_path, dpi=600)
        return img_path


if __name__ == "__main__":
    args = parse_args()
    nas: DyNAS = DyNAS(
        {
            "search_algorithm": args.search_algorithm,
            "supernet": args.supernet,
            "seed": args.seed,
            "metrics": args.metrics,
            "population": args.population,
            "num_evals": args.num_evals,
            "results_csv_path": args.results_csv_path,
            "batch_size": args.batch_size,
            "dataset_path": args.dataset_path,
        },
    )
    nas(os.path.join(args.work_path, args.img_file_name))
