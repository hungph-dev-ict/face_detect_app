import * as cv from "opencv4nodejs";
import {
  getModelFilePath
} from "./utils/commons";

import * as path from 'path';
import * as fs from 'fs';
import * as readline from 'readline';
import {
  trainingNewModel
} from './utils/training';
import {
  createImageData
} from './utils/createImage';
const fr = require("face-recognition").withCv(cv);

const trainedModelFile = "faceRecognitionModel_HungPH.json";
const trainedModelFilePath = path.resolve(getModelFilePath(), trainedModelFile);
const recognizer = fr.FaceRecognizer();

if (!fs.existsSync(trainedModelFilePath)) {
  throw new Error(
    "[Notification] Model file not found, please run the training file first to train and save the model."
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

let varsTimeIn = new Array();
let varsTimeOut = new Array();

function runVideoFaceDetection(src, detectFaces) {
  grabFrames(src, 10, frame => {
    let flagsDetect = new Array();
    const frameResized = frame.resizeToMax(800);
    const threshold = 0.8;
    // detect faces
    const faceRects = detectFaces(frameResized);
    if (faceRects.length) {
      faceRects.forEach(faceRect => {
        const {
          rect,
          face
        } = faceRect;
        const cvFace = fr.CvImage(face);
        const prediction = recognizer.predictBest(cvFace, threshold);
        const text = `${prediction.className} (${prediction.distance})`;
        const blue = new cv.Vec(255, 0, 0);
        // draw Rect
        drawRectWithText(frameResized, faceRect.rect, text, blue);
        // log timestamp
        if (!varsTimeIn[prediction.className]) {
          varsTimeIn[prediction.className] = new Date();
        }
        if (varsTimeOut[prediction.className]) {
          varsTimeOut[prediction.className] = 0;
        }
        flagsDetect[prediction.className] = 1;
      });
    } else {
      for (let className in varsTimeOut) {
        if (varsTimeOut.hasOwnProperty(className)) {
          if (varsTimeOut[className]) {
            let seconds = (new Date().getTime() - varsTimeOut[className].getTime()) / 1000;
            if (seconds > 10 && className != 'unknown') {
              if ((varsTimeOut[className].getTime() - varsTimeIn[className].getTime()) / 1000 > 10) {
                console.log(
                  "* ---|------------|--------------------------------|---------------------------- *"
                );
                console.log(
                  '* 1  |' + addSpaceToEndString(className, 12) + '|      ' +
                  varsTimeIn[className].toISOString()
                  .replace(/T/, " ")
                  .replace(/\..+/, "") +
                  "       |     " +
                  varsTimeOut[className].toISOString()
                  .replace(/T/, " ")
                  .replace(/\..+/, "") +
                  "     *"
                );
                varsTimeIn[className] = 0;
                varsTimeOut[className] = 0;
              } else {
                varsTimeIn[className] = 0;
                varsTimeOut[className] = 0;
              }
            }
          }
        }
      }
    }
    for (let className in varsTimeIn) {
      if (varsTimeIn.hasOwnProperty(className)) {
        if (!flagsDetect[className] && varsTimeIn[className] && !varsTimeOut[className]) {
          varsTimeOut[className] = new Date();
        }
      }
    }
    cv.imshow("Face Detection Application by HungPhamHoang", frameResized);
    cv.waitKey(1);
  });
}

function grabFrames(videoFile, delay, onFrame) {
  const wCap = new cv.VideoCapture(videoFile);
  let done = false;
  // open capture from webcam
  console.log("[Notification] Start Streaming Face Recognition \n");
  console.log(
    "**********************************************************************************"
  );
  console.log(
    "* ID | Human Name |           Time Start           |           Time End          *"
  );
  const intvl = setInterval(() => {
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
      clearInterval(intvl);
      console.log(
        "********************************************************************************** \n"
      );
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

const rdln = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rdln.question('Please choose: \n 1. Run webcam detection \n 2. Run new face training \n Your choice: ', answer => {
  if (answer === '1') {
    runVideoFaceDetection(webcamPort, detectFaces);
    cv.waitKey();
    rdln.close();
  } else if (answer === '2') {
    runVideoNewFaceTraining(webcamPort, detectFaces);
    cv.waitKey();
  }
})

let total = 0;
let check_face;

function runVideoNewFaceTraining(src, detectFaces) {
  rdln.question('[Notification] Please see camera to create your face data, type your name: ', answer_face_name => {
    grabFramesTraining(src, 10, frame => {
      check_face = false;
      const frameResized = frame.resizeToMax(800);
      const threshold = 0.8;
      // detect faces
      const faceRects = detectFaces(frameResized);
      if (faceRects.length) {
        if (faceRects.length == true) {
          check_face = true;
        }
        faceRects.forEach(faceRect => {
          const {
            rect,
            face
          } = faceRect;
          const cvFace = fr.CvImage(face);
          const prediction = recognizer.predictBest(cvFace, threshold);
          const text = `${prediction.className} (${prediction.distance})`;
          const blue = new cv.Vec(255, 0, 0);
          // draw Rect
          drawRectWithText(frameResized, faceRect.rect, text, blue);
        });
      }
      cv.imshow("Face Detection Application by Hung Pham Hoang", frameResized);
      cv.waitKey(1);
    }, answer_face_name);
  })
}

function grabFramesTraining(videoFile, delay, onFrame, answer_face_name) {
  let done = false;
  const wCap = new cv.VideoCapture(videoFile);
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
    if (done && check_face) {
      createImageData(answer_face_name, total, frame);
      console.log('Image ' + total + ' is saved !');
      total++;
    }
    if (total > 10) {
      clearInterval(interval);
      console.log("[Notification] End Streaming Face Recognition \n");
      trainingNewModel([answer_face_name]);
      rdln.close();
    }
  }, 0);
};

function addSpaceToEndString(string, length) {
  var currentLength = string.length;
  var spaceCharacterCounting = length - currentLength;
  for(var i = 0; i < spaceCharacterCounting; i++) {
    string += ' ';
  }

  return string;
}