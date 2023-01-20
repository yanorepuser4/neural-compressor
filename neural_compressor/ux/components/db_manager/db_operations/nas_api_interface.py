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
# pylint: disable=no-member
"""INC Bench Project API interface."""
import os
import shutil

from sqlalchemy.orm import sessionmaker
from werkzeug.wrappers import Response

from neural_compressor.ux.components.db_manager.db_manager import DBManager
from neural_compressor.ux.components.db_manager.db_models.nas import NAS
from neural_compressor.ux.utils.consts import WORKSPACE_LOCATION, ExecutionStatus
from neural_compressor.ux.utils.exceptions import ClientErrorException

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)


class NasAPIInterface:
    """Interface for queries connected with NAS."""

    @staticmethod
    def create_nas(data: dict) -> dict:
        """Create new nas project and add input model."""
        if data.get("name", None) is None:
            ClientErrorException("nas name not provided.")
        if data.get("csv_file_name", None) is None:
            ClientErrorException("csv_file_name not provided.")
        if data.get("img_file_name", None) is None:
            ClientErrorException("img_file_name not provided.")

        if not os.path.exists(os.path.join(WORKSPACE_LOCATION, "NAS")):
            os.mkdir(os.path.join(WORKSPACE_LOCATION, "NAS"))

        with Session.begin() as db_session:
            nas_id: int = NAS.add(db_session, data)
            data.update({"nas_id": nas_id})

        if not os.path.exists(os.path.join(WORKSPACE_LOCATION, "NAS", str(nas_id))):
            os.mkdir(os.path.join(WORKSPACE_LOCATION, "NAS", str(nas_id)))

        return {
            "nas_id": nas_id,
        }

    @staticmethod
    def delete_nas(data: dict) -> dict:
        """Delete nas project from database and workspace."""
        try:
            nas_id: int = int(data.get("id", None))
            nas_project_name: str = str(data.get("name", None))
        except ValueError:
            raise ClientErrorException("Could not parse value.")
        except TypeError:
            raise ClientErrorException("Missing project id or project name.")
        with Session.begin() as db_session:
            removed_nas_id = NAS.delete(
                db_session=db_session,
                nas_id=nas_id,
                nas_project_name=nas_project_name,
            )

        if removed_nas_id is not None:
            nas_project_save_location = os.path.join(
                WORKSPACE_LOCATION,
                "NAS",
                str(removed_nas_id),
            )
            shutil.rmtree(nas_project_save_location, ignore_errors=True)

        return {"id": removed_nas_id}

    @staticmethod
    def get_nas_details(data: dict) -> dict:
        """Get nas details from database."""
        try:
            nas_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect nas id.")
        except TypeError:
            raise ClientErrorException("Could not find nas id.")
        with Session.begin() as db_session:
            nas_details = NAS.details(
                db_session=db_session,
                nas_id=nas_id,
            )
        return nas_details

    @staticmethod
    def update_nas_status(data: dict) -> dict:
        """Update nas status."""
        try:
            nas_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect nas id.")
        except TypeError:
            raise ClientErrorException("Could not find nas id.")

        try:
            status: ExecutionStatus = ExecutionStatus(data.get("status", None))
        except ValueError as err:
            raise ClientErrorException(err)

        with Session.begin() as db_session:
            response_data = NAS.update_status(
                db_session,
                nas_id,
                status,
            )
        return response_data

    @staticmethod
    def list_nas(data: dict) -> dict:
        """List nas assigned to project."""
        with Session.begin() as db_session:
            nas_job_list = NAS.list(
                db_session,
            )
        return nas_job_list

    @staticmethod
    def show_image(data: dict) -> Response:
        """Send image from database."""
        try:
            nas_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect nas id.")
        except TypeError:
            raise ClientErrorException("Could not find nas id.")

        work_path = os.path.join(WORKSPACE_LOCATION, "NAS")
        with Session.begin() as db_session:
            result = NAS.get_img(
                db_session=db_session,
                nas_id=nas_id,
                work_path=work_path,
            )
        return result
