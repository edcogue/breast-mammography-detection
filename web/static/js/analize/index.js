/*  ==========================================
          SHOW UPLOADED IMAGE AND UPLOAD IT
        * ========================================== */

var app;
function readURL(input) {
  if (input.files && input.files[0]) {
    app.reset();
    app.init({
      dataViewConfigs: { "*": [{ divId: "layerGroup0" }] },
    });
    app.loadFiles(input.files);
  }
}

$(function () {
  $("#upload").on("change", function () {
    $("#alert_banner").hide();
    $(".image-area").addClass("hide-bg");
    readURL(input);
  });
});

$(document).ready(function (e) {
  app = new dwv.App();
  app.init({
    dataViewConfigs: { "*": [{ divId: "layerGroup0" }] },
  });
  $("#refresh_button").on("click", function () {
    window.location.reload();
  });
  $("#submit_button").on("click", function () {
    $("#alert_banner").hide();
    $("#loader_spiner").show();
    var file_data = $("#upload").prop("files")[0];
    var form_data = new FormData();
    form_data.append("file", file_data);
    $.ajax({
      url: base_url,
      dataType: "text",
      cache: false,
      contentType: false,
      processData: false,
      data: form_data,
      type: "post",
      success: function (response) {
        $("#notPreview").hide();
        app.reset();
        app.init({
          dataViewConfigs: { "*": [{ divId: "layerGroup0" }] },
        });
        urltoFile(response, "result.png", "image/png").then(function (file) {
          app.loadFiles([file]);
        });

        $("#download_link").prop("href", response);
        $("#loader_spiner").hide();
        $("#upload_image_container").hide();
        $("#submit_button").hide();
        $("#download_button").show();
        $("#legend").show();
        $("#refresh_button").show();
      },
      error: function (response) {
        $("#alert_banner").html(response.responseText);
        $("#loader_spiner").hide();
        $("#legend").hide();
        $("#alert_banner").show();
        $("#notPreview").hide();
      },
    });
  });
});

/*  ==========================================
              SHOW UPLOADED IMAGE NAME
            * ========================================== */
var input = document.getElementById("upload");
var infoArea = document.getElementById("upload-label");

input.addEventListener("change", showFileName);
function showFileName(event) {
  var input = event.srcElement;
  var fileName = input.files[0].name;
  infoArea.textContent = "File name: " + fileName;
}

function urltoFile(url, filename, mimeType) {
  return fetch(url)
    .then(function (res) {
      return res.arrayBuffer();
    })
    .then(function (buf) {
      return new File([buf], filename, { type: mimeType });
    });
}
