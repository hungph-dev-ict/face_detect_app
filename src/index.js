import * as cv from "opencv4nodejs";
import {
  getModelFilePath
} from "./utils/commons";

import * as path from 'path';
import * as fs from 'fs';
const fr = require("face-recognition").withCv(cv);

const trainedModelFile = "faceRecognitionModel_HungPH.json";
const trainedModelFilePath = path.resolve(getModelFilePath(), trainedModelFile);
const recognizer = fr.FaceRecognizer();

if (!fs.existsSync(trainedModelFilePath)) {
  throw new Error(
    "model file not found, please run the faceRecognition2 example first to train and save the model"
  );
} else {
  recognizer.load(require(trainedModelFilePath));
}

function drawRectWithText(image, rect, text, color) {
  const thickness = 2;
  image.drawRectangle(
    new cv.Point(rect.x, rect.y),
    new cv.Point(rect.x + rect.width, rect.y + rect.height),
    color,
    cv.LINE_8,
    thickness
  );

  const textOffsetY = rect.height + 20;
  image.putText(
    text,
    new cv.Point(rect.x, rect.y + textOffsetY),
    cv.FONT_ITALIC,
    0.6,
    color,
    thickness
  );
}

function runVideoFaceDetection(src, detectFaces) {
  grabFrames(src, 10, frame => {
    const frameResized = frame.resizeToMax(800);
    // detect faces
    const faceRects = detectFaces(frameResized);
    if (faceRects.length) {
      faceRects.forEach(faceRect => {
        const {
          rect,
          face
        } = faceRect;
        const cvFace = fr.CvImage(face);
        const prediction = recognizer.predictBest(cvFace);
        const text = `${prediction.className} (${prediction.distance})`;
        const blue = new cv.Vec(255, 0, 0);
        // draw Rect
        drawRectWithText(frameResized, faceRect.rect, text, blue);
      });
    }
    cv.imshow("Face Detection Application by HungPhamHoang", frameResized);
    cv.waitKey(1);
  });
}


function grabFrames(videoFile, delay, onFrame) {
  const wCap = new cv.VideoCapture(videoFile);
  let done = false;
  const interval = setInterval(() => {
    let frame = wCap.read();
    // loop back to start on end of stream reached
    if (frame.empty) {
      wCap.reset();
      frame = wCap.read();
    }
    onFrame(frame);

    const key = cv.waitKey(delay);
    done = key !== -1 && key !== 255;
    if (done) {
      clearInterval(interval);
      console.log("[Notification] End Streaming Face Recognition \n");
      console.log("[Event] Key pressed, exiting.");
    }
  }, 0);
};

const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);

const webcamPort = 0;

function detectFaces(img) {
  const options = {
    minSize: new cv.Size(100, 100),
    scaleFactor: 1.2,
    minNeighbors: 10
  };
  const {
    objects,
    map
  } = classifier.detectMultiScale(
    img.bgrToGray(),
    options
  );

  return objects.map(rect => ({
    rect,
    face: img.getRegion(rect).copy()
  }));
}

runVideoFaceDetection(webcamPort, detectFaces);
cv.waitKey();