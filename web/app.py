import os
import flask
import errors
import cv2

from aux_functions import *
import detr_detector as detr
import yolo_detector as yolo

# APP SERVER
app = flask.Flask(__name__)


@app.route("/")
def index():
    return flask.render_template("index.html")


@app.route("/analize_vit")
def analize_vit_view():
    return flask.render_template("base_view.html", title="Detección de anomalias con ViT", post_endpoint="/analize_vit", legend=True)


@app.route("/analize_vit", methods=["POST"])
def analize_vit_image():
    if "file" not in flask.request.files:
        return errors.ERROR_NOT_FILE_UPLOAD, 400
    file = flask.request.files["file"]
    if file.filename == "":
        return errors.ERROR_NOT_FILE_UPLOAD, 400
    if not allowed_file(file.filename):
        return errors.ERROR_EXTENSION_NOT_ALLOWED, 400

    # Save uploaded image
    upload_file_path, result_file_path = save_file(file)

    # Setup post-exectute to try to delete both files just after sending response
    @flask.after_this_request
    def add_close_action(response):
        @response.call_on_close
        def process_after_request():
            delete_file(upload_file_path)
            delete_file(result_file_path)
        return response

    ################# ANALIZAR ####################################
    image = cv2.imread(upload_file_path)
    image = resize_pad_and_clahe(image)
    image = denoise(image)
    image = detr.detect_abnormalities(image)
    image.save(result_file_path)
    exit_code = 0
    # exit_code, result_file_path = process_image(
    #     upload_file_path, result_file_path)

    if exit_code != 0:  # Exit with code ! = 0 means error on command exectuion
        print(errors.ERROR_PROCESSING)
        return errors.ERROR_PROCESSING, 400

    ##################################################################

    # Read and encode processed file for response
    response = encode_result_file(result_file_path)
    if response:
        return response
    else:
        return errors.ERROR_PROCESSING, 400


@app.route("/analize_yolo")
def analize_yolo_view():
    return flask.render_template("base_view.html", title="Detección de anomalias con YOLOv8", post_endpoint="/analize_yolo", legend=True)


@app.route("/analize_yolo", methods=["POST"])
def analize_yolo_image():
    if "file" not in flask.request.files:
        return errors.ERROR_NOT_FILE_UPLOAD, 400
    file = flask.request.files["file"]
    if file.filename == "":
        return errors.ERROR_NOT_FILE_UPLOAD, 400
    if not allowed_file(file.filename):
        return errors.ERROR_EXTENSION_NOT_ALLOWED, 400

    # Save uploaded image
    upload_file_path, result_file_path = save_file(file)

    # Setup post-exectute to try to delete both files just after sending response
    @flask.after_this_request
    def add_close_action(response):
        @response.call_on_close
        def process_after_request():
            delete_file(upload_file_path)
            delete_file(result_file_path)
        return response

    ################# ANALIZAR ####################################
    image = cv2.imread(upload_file_path)
    image = resize_pad_and_clahe(image)
    image = denoise(image)
    image = yolo.detect_abnormalities(image)
    image.save(result_file_path)
    exit_code = 0
    # exit_code, result_file_path = process_image(
    #     upload_file_path, result_file_path)

    if exit_code != 0:  # Exit with code ! = 0 means error on command exectuion
        print(errors.ERROR_PROCESSING)
        return errors.ERROR_PROCESSING, 400

    ##################################################################

    # Read and encode processed file for response
    response = encode_result_file(result_file_path)
    if response:
        return response
    else:
        return errors.ERROR_PROCESSING, 400


@app.route("/denoise")
def denoise_view():
    return flask.render_template("base_view.html", title="Supresión de ruido para mamografías", post_endpoint="/denoise", is_denoiser=True, legend=False)


@app.route("/denoise", methods=["POST"])
def denoise_image():
    if "file" not in flask.request.files:
        return errors.ERROR_NOT_FILE_UPLOAD, 400
    file = flask.request.files["file"]
    if file.filename == "":
        return errors.ERROR_NOT_FILE_UPLOAD, 400
    if not allowed_file(file.filename):
        return errors.ERROR_EXTENSION_NOT_ALLOWED, 400

    # Save uploaded image
    upload_file_path, result_file_path = save_file(file)

    # Setup post-exectute to try to delete both files just after sending response
    @flask.after_this_request
    def add_close_action(response):
        @response.call_on_close
        def process_after_request():
            delete_file(upload_file_path)
            delete_file(result_file_path)
        return response

    ################# DENOISE ####################################
    try:
        image = cv2.imread(upload_file_path)
        image = resize_pad_and_clahe(image)
        image = denoise(image)
        cv2.imwrite(result_file_path, image)
        exit_code = 0
    except:
        exit_code = 1

    if exit_code != 0:  # Exit with code ! = 0 means error on command exectuion
        print(errors.ERROR_PROCESSING)
        return errors.ERROR_PROCESSING, 400

    ##################################################################

    # Read and encode processed file for response
    response = encode_result_file(result_file_path)
    if response:
        return response
    else:
        return errors.ERROR_PROCESSING, 400



if app.debug:
    port = int(os.environ.get("PORT", 8888))
    app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
