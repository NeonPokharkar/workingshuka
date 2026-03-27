import os

import static.constants as constants

def get_notes_list():
    # Get all entries (files and folders)
    entries = os.listdir(constants.RAG_PATH)

    # Filter for only files (optional, as listdir returns everything)
    file_names = [entry for entry in entries if os.path.isfile(os.path.join(constants.RAG_PATH, entry))]

    return file_names