import pandas as pd
import numpy as np
import os
from typing import List
from fastapi import HTTPException
import logging
from .custom_api_responses import API_STATUS_CODE


def add_embedding(
    db_filepath: str,
    subject_id: str,
    embedding: np.ndarray,
):
    """Add embedding into db csv file. If give db file path does not exist,
    new file will be created.

    Parameters
    ----------
    db_filepath : str
    subject_id : str
    embedding : array with shape (m,), same as other embedding in db
    """
    embedding = embedding.astype(float)  # ensure data type
    if not os.path.exists(db_filepath):
        subject_arr = np.array([subject_id])
        db = pd.DataFrame(np.concatenate((subject_arr, embedding))[np.newaxis, ...])
        db.columns = ["subject_id"] + [i for i in range(len(embedding))]
        db.to_csv(db_filepath, index=False)
    else:
        db = pd.read_csv(db_filepath)
        assert (db.shape[1] - 1) == embedding.shape[0], (
            f"Embedding vector ({embedding.shape[0]},) does not share "
            + f"the same length with others in db ({db.shape[1] - 1},)."
        )
        subject_arr = np.array([subject_id])
        db.loc[len(db)] = np.concatenate((subject_arr, embedding)).ravel()
        db.to_csv(db_filepath, index=False)
    
    del db


def get_db(db_filepath: str):
    if not os.path.exists(db_filepath):
        error_message = "No db file is found. You may use /recognize/db/add first."
        logging.error(error_message)
        raise HTTPException(
            status_code=API_STATUS_CODE["NO_DB_FILE_FOUND"],
            detail=error_message,
        )
    db = pd.read_csv(db_filepath)
    return db


def query_subjects(db_filepath: str) -> List[str]:
    """Query all subjects in db.

    Parameters
    ----------
    db_filepath : str

    Returns
    -------
    List[str]
        List of subject ids
    """
    db = get_db(db_filepath)
    subjects = db["subject_id"].to_list()
    del db
    return subjects


def remove_subject(db_filepath: str, subject_id: str):
    """Remove all embeddings with a given subject id.

    Parameters
    ----------
    db_filepath : str
    subject_id : str
    """
    db = get_db(db_filepath)
    if subject_id not in db["subject_id"].values:
        error_message = (
            f"There is not embedding with a given subject id {subject_id} in db."
        )
        logging.error(error_message)
        raise HTTPException(
            status_code=API_STATUS_CODE["NO_SUBJECT_ID_IN_DB"],
            detail=error_message,
        )
    db = db[db["subject_id"] != subject_id]
    db.to_csv(db_filepath, index=False)
    del db
