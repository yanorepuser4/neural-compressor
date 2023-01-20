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
"""INC Bench NAS."""

import os
from typing import Any, Optional

from flask import send_file
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.orm import session
from sqlalchemy.sql import func
from werkzeug.wrappers import Response

from neural_compressor.ux.components.db_manager.db_manager import Base
from neural_compressor.ux.utils.consts import ExecutionStatus


class NAS(Base):
    """INC Bench NAS class."""

    __tablename__ = "NAS"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    notes = Column(String(250), nullable=True)
    csv_file_name = Column(String(250), nullable=False)
    img_file_name = Column(String(250), nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    modified_at = Column(DateTime, nullable=True, onupdate=func.now())
    search_algorithm = Column(String(50), nullable=False)
    supernet = Column(String(250), nullable=False)
    seed = Column(Integer, nullable=False)
    metrics = Column(String(50), nullable=False)
    population = Column(Integer, nullable=False)
    num_evals = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    dataset_path = Column(String(250), nullable=False)
    status = Column(String(50), nullable=True)

    @staticmethod
    def add(db_session: session.Session, data: dict) -> int:
        """
        Create nas project object.

        returns nas id of created nas project
        """
        new_nas_project = NAS(
            name=data.get("name"),
            csv_file_name=data.get("csv_file_name"),
            img_file_name=data.get("img_file_name"),
            search_algorithm=data.get("search_algorithm"),
            supernet=data.get("supernet"),
            seed=data.get("seed"),
            metrics=data.get("metrics"),
            population=data.get("population"),
            num_evals=data.get("num_evals"),
            batch_size=data.get("batch_size"),
            dataset_path=data.get("dataset_path"),
        )
        db_session.add(new_nas_project)
        db_session.flush()
        return int(new_nas_project.id)

    @staticmethod
    def delete(
        db_session: session.Session,
        nas_id: int,
        nas_project_name: str,
    ) -> Optional[int]:
        """Remove nas project from database and workdir."""
        nas_project = (
            db_session.query(NAS)
            .filter(NAS.id == nas_id)
            .filter(NAS.name == nas_project_name)
            .one_or_none()
        )
        if nas_project is None:
            return None
        db_session.delete(nas_project)
        db_session.flush()

        return int(nas_project.id)

    @staticmethod
    def details(db_session: session.Session, nas_id: int) -> dict:
        """Get nas details from database."""
        nas_project = db_session.query(NAS).filter(NAS.id == nas_id)[0]

        return {
            "id": nas_project.id,
            "name": nas_project.name,
            "notes": nas_project.notes,
            "csv_file_name": nas_project.csv_file_name,
            "img_file_name": nas_project.img_file_name,
            "created_at": str(nas_project.created_at),
            "modified_at": str(nas_project.modified_at),
            "search_algorithm": nas_project.search_algorithm,
            "supernet": nas_project.supernet,
            "seed": nas_project.seed,
            "metrics": nas_project.metrics,
            "population": nas_project.population,
            "num_evals": nas_project.num_evals,
            "batch_size": nas_project.batch_size,
            "dataset_path": nas_project.dataset_path,
            "status": nas_project.status,
        }

    @staticmethod
    def update_status(
        db_session: session.Session,
        nas_id: int,
        execution_status: ExecutionStatus,
    ) -> dict:
        """Update nas status."""
        nas = db_session.query(NAS).filter(NAS.id == nas_id).one()
        nas.status = execution_status.value
        db_session.add(nas)
        db_session.flush()
        return {
            "id": nas.id,
            "status": nas.status,
        }

    @staticmethod
    def list(db_session: session.Session) -> dict:
        """Get nas list from database."""
        nas_instances = db_session.query(NAS).all()
        nas = [nas_info.build_info(nas=nas_info) for nas_info in nas_instances]
        return {"nas": nas}

    @staticmethod
    def build_info(
        nas: Any,
    ) -> dict:
        """Build nas info."""
        nas_info = {
            "id": nas.id,
            "name": nas.name,
            "notes": nas.notes,
            "csv_file_name": nas.csv_file_name,
            "img_file_name": nas.img_file_name,
            "created_at": str(nas.created_at),
            "modified_at": str(nas.modified_at),
            "search_algorithm": nas.search_algorithm,
            "supernet": nas.supernet,
            "seed": nas.seed,
            "metrics": nas.metrics,
            "population": nas.population,
            "num_evals": nas.num_evals,
            "batch_size": nas.batch_size,
            "dataset_path": nas.dataset_path,
            "status": nas.status,
        }
        return nas_info

    @staticmethod
    def get_img(db_session: session.Session, nas_id: int, work_path: str) -> Response:
        """Send image from database."""
        nas_project = db_session.query(NAS).filter(NAS.id == nas_id)[0]
        img_file_path = os.path.join(work_path, str(nas_id), nas_project.img_file_name)
        return send_file(
            img_file_path,
            mimetype="image/png",
            as_attachment=False,
        )
