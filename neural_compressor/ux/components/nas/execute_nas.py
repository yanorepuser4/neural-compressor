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
"""Execute nas."""

import os
from typing import Any, Dict

from neural_compressor.ux.components.db_manager.db_operations.nas_api_interface import (
    NasAPIInterface,
)
from neural_compressor.ux.utils.consts import WORKSPACE_LOCATION, ExecutionStatus
from neural_compressor.ux.utils.exceptions import InternalException
from neural_compressor.ux.utils.executor import Executor
from neural_compressor.ux.web.communication import MessageQueue

mq = MessageQueue()


def execute_nas(nas_info: Dict[str, Any]) -> None:
    """
    Execute nas.

    Expected data:
    {
        "nas_id": "1",
        "request_id": "asd",
    }
    """
    details = NasAPIInterface.get_nas_details({"id": nas_info.get("nas_id")})
    details["work_path"] = os.path.join(WORKSPACE_LOCATION, "NAS", str(details.get("id")))
    details["results_csv_path"] = os.path.join(
        str(details.get("work_path")),
        str(details.get("csv_file_name")),
    )

    request_id: str = str(nas_info["request_id"])

    try:
        execute_real_nas(
            request_id=request_id,
            nas_details=details,
        )

        mq.post_success(
            "execute_finish",
            {
                "request_id": request_id,
                "id": nas_info.get("nas_id"),
            },
        )
    except Exception:
        NasAPIInterface.update_nas_status(
            {
                "id": nas_info.get("nas_id"),
                "status": ExecutionStatus.ERROR,
            },
        )
        mq.post_error(
            "nas_finish",
            {"message": "Failure", "code": 404, "request_id": request_id},
        )
        raise


def execute_real_nas(
    request_id: str,
    nas_details: dict,
) -> dict:
    """Execute NAS."""
    script = os.path.join(
        os.path.dirname(__file__),
        "nas.py",
    )
    logs = [os.path.join(str(nas_details.get("work_path")), "output.txt")]
    metrics = ["acc", "macs"] if (nas_details.get("metrics") == "acc_macs") else ["acc", "lat"]

    command = [
        "python",
        script,
        "--search_algorithm",
        nas_details.get("search_algorithm"),
        "--supernet",
        nas_details.get("supernet"),
        "--seed",
        nas_details.get("seed"),
        "--metrics",
        metrics[0],
        metrics[1],
        "--population",
        nas_details.get("population"),
        "--num_evals",
        nas_details.get("num_evals"),
        "--results_csv_path",
        nas_details.get("results_csv_path"),
        "--batch_size",
        nas_details.get("batch_size"),
        "--dataset_path",
        nas_details.get("dataset_path"),
        "--work_path",
        nas_details.get("work_path"),
        "--img_file_name",
        nas_details.get("img_file_name"),
    ]
    send_data = {
        "message": "started",
        "request_id": request_id,
    }
    executor = Executor(
        workspace_path=str(nas_details.get("work_path")),
        subject="nas",
        data=send_data,
        log_name="output",
    )
    proc = executor.call(command)
    if not proc.is_ok:
        raise InternalException("NAS failed during execution.")

    NasAPIInterface.update_nas_status(
        {
            "id": nas_details.get("id"),
            "status": ExecutionStatus.SUCCESS,
        },
    )
    response_data = {
        "request_id": request_id,
        "log_path": logs,
    }

    return response_data
