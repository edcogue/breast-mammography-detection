import os
import sys
import base64
import mimetypes
import time
import errors
import glob
from werkzeug.utils import secure_filename
from preprocess_image import *

ALLOWED_EXTENSIONS = {"png", "jpeg", "jpg"}

UPLOAD_FOLDER = os.path.join("/app", "temp_images")
RESULT_FOLDER = os.path.join("/app/static/images/", "analized")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def delete_file(path):
    # Delete uploaded and resulted file
    try:
        path_no_extension = os.path.splitext(path)[0]
        for file in glob.glob(path_no_extension + ".*"):
            os.remove(file)
    except OSError as e:
        print(errors.ERROR_DELETING_FILE, e)


def allowed_file(filename):
    # Check for allowed file extension on uploaded image
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_filemame(_file):
    # Generate file name with {{timestamp.extension}} format
    ts_milliseconds = int(round(time.time() * 1000))
    file_extension = os.path.splitext(_file.filename)[1]
    file_name = secure_filename(str(ts_milliseconds)+file_extension)
    return file_name


def save_file(file):
    # Save file on UPLOADS folder and provide the paths for files
    file_name = generate_filemame(file)
    upload_file_path = os.path.join(UPLOAD_FOLDER, file_name)
    file.save(upload_file_path)
    result_file_path = os.path.join(RESULT_FOLDER, file_name)
    return upload_file_path, result_file_path


def encode_result_file(result_file_path):
    # Encode result image file to BASE64 so the files can be deleted
    # after response without aftecting user
    if (os.path.exists(result_file_path)):
        try:
            file_mimetype = mimetypes.guess_type(result_file_path)[0]
            result_file = open(result_file_path, "rb")
            encoded_image = base64.b64encode(result_file.read())
            image_src = "data:" + file_mimetype + \
                ";base64," + encoded_image.decode("utf8")
            return image_src
        except Exception as e:
            print(errors.ERROR_FILE_ENCODING, e)
            return False
    else:
        print(errors.ERROR_FILE_NOT_FOUND)
        return False
