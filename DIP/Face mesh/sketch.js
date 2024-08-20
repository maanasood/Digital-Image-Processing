
console.log(drawing);

let x = [];
let y = [];
let fourierX;
let fourierY;
let time = 0;
let path = [];

let fourierlipsX;
let fourierlipsY;
let fouriernoseX;
let fouriernoseY;
let fourier_left_eye_X;
let fourier_left_eye_Y;
let fourier_right_eye_X;
let fourier_right_eye_Y;

let fourier_left_eyebrow_X;
let fourier_left_eyebrow_Y;
let fourier_left_eye_outer_X;
let fourier_left_eye_outer_Y;
let fourier_left_eye_inner_X;
let fourier_left_eye_inner_Y;

let fourier_right_eyebrow_X;
let fourier_right_eyebrow_Y;
let fourier_right_eye_outer_X;
let fourier_right_eye_outer_Y;
let fourier_right_eye_inner_X;
let fourier_right_eye_inner_Y;

let fourier_face_X;
let fourier_face_Y;

let lips_time = 0;
let nose_time = 0;
let left_eye_time = 0;
let right_eye_time = 0;

let left_eyebrow_time = 0;
let left_eye_outer_time = 0;
let left_eye_inner_time = 0;

let right_eyebrow_time = 0;
let right_eye_outer_time = 0;
let right_eye_inner_time = 0;

let face_time = 0;

let lipsX = [];
let lipsY = [];
let noseX = [];
let noseY = [];
let left_eye_x = [];
let left_eye_y = [];
let right_eye_x = [];
let right_eye_y = [];

let left_eyebrow_x = [];
let left_eyebrow_y = [];
let left_eye_outer_x = [];
let left_eye_outer_y = [];
let left_eye_inner_x = [];
let left_eye_inner_y = [];

let right_eyebrow_x = [];
let right_eyebrow_y = [];
let right_eye_outer_x = [];
let right_eye_outer_y = [];
let right_eye_inner_x = [];
let right_eye_inner_y = [];

let face_x = [];
let face_y = [];

let lipsPath = [];
let nosePath = [];
let left_eye_path = [];
let right_eye_path = [];

let left_eyebrow_path = [];
let left_eye_outer_path = [];
let left_eye_inner_path = [];

let right_eyebrow_path = [];
let right_eye_outer_path = [];
let right_eye_inner_path = [];

let face_path = [];

function creating_lips() {
  const skip = 1;
  for (let i = 0; i < lips.length; i += skip) {
    lipsX.push(lips[i].x * 3);
    lipsY.push(lips[i].y * 3);
  }
  fourierlipsX = dft(lipsX);
  fourierlipsY = dft(lipsY);

  fourierlipsX.sort((a, b) => b.amp - a.amp);
  fourierlipsY.sort((a, b) => b.amp - a.amp);
}

function creating_nose() {
  const skip = 1;
  for (let i = 0; i < nose.length; i += skip) {
    noseX.push(nose[i].x * 3);
    noseY.push(nose[i].y * 3);
  }
  fouriernoseX = dft(noseX);
  fouriernoseY = dft(noseY);

  fouriernoseX.sort((a, b) => b.amp - a.amp);
  fouriernoseY.sort((a, b) => b.amp - a.amp);
}

// LEFT SIDE
function creating_left_eye() {
  const skip = 1;
  for (let i = 0; i < left_eye.length; i += skip) {
    left_eye_x.push(left_eye[i].x * 3);
    left_eye_y.push(left_eye[i].y * 3);
  }
  fourier_left_eye_X = dft(left_eye_x);
  fourier_left_eye_Y = dft(left_eye_y);

  fourier_left_eye_X.sort((a, b) => b.amp - a.amp);
  fourier_left_eye_Y.sort((a, b) => b.amp - a.amp);
}

function creating_left_eyebrow() {
  const skip = 1;
  for (let i = 0; i < left_eyebrow.length; i += skip) {
    left_eyebrow_x.push(left_eyebrow[i].x * 3);
    left_eyebrow_y.push(left_eyebrow[i].y * 3);
  }
  fourier_left_eyebrow_X = dft(left_eyebrow_x);
  fourier_left_eyebrow_Y = dft(left_eyebrow_y);

  fourier_left_eyebrow_X.sort((a, b) => b.amp - a.amp);
  fourier_left_eyebrow_Y.sort((a, b) => b.amp - a.amp);
}

function creating_left_eye_outer() {
  const skip = 1;
  for (let i = 0; i < left_eye_outer.length; i += skip) {
    left_eye_outer_x.push(left_eye_outer[i].x * 3);
    left_eye_outer_y.push(left_eye_outer[i].y * 3);
  }
  fourier_left_eye_outer_X = dft(left_eye_outer_x);
  fourier_left_eye_outer_Y = dft(left_eye_outer_y);

  fourier_left_eye_outer_X.sort((a, b) => b.amp - a.amp);
  fourier_left_eye_outer_Y.sort((a, b) => b.amp - a.amp);
}

function creating_left_eye_inner() {
  const skip = 1;
  for (let i = 0; i < left_eye_inner.length; i += skip) {
    left_eye_inner_x.push(left_eye_inner[i].x * 3);
    left_eye_inner_y.push(left_eye_inner[i].y * 3);
  }
  fourier_left_eye_inner_X = dft(left_eye_inner_x);
  fourier_left_eye_inner_Y = dft(left_eye_inner_y);

  fourier_left_eye_inner_X.sort((a, b) => b.amp - a.amp);
  fourier_left_eye_inner_Y.sort((a, b) => b.amp - a.amp);
}

// RIGHT SIDE
function creating_right_eye() {
  const skip = 1;
  for (let i = 0; i < right_eye.length; i += skip) {
    right_eye_x.push(right_eye[i].x * 3);
    right_eye_y.push(right_eye[i].y * 3);
  }
  fourier_right_eye_X = dft(right_eye_x);
  fourier_right_eye_Y = dft(right_eye_y);

  fourier_right_eye_X.sort((a, b) => b.amp - a.amp);
  fourier_right_eye_Y.sort((a, b) => b.amp - a.amp);
}

function creating_right_eyebrow() {
  const skip = 1;
  for (let i = 0; i < right_eyebrow.length; i += skip) {
    right_eyebrow_x.push(right_eyebrow[i].x * 3);
    right_eyebrow_y.push(right_eyebrow[i].y * 3);
  }
  fourier_right_eyebrow_X = dft(right_eyebrow_x);
  fourier_right_eyebrow_Y = dft(right_eyebrow_y);

  fourier_right_eyebrow_X.sort((a, b) => b.amp - a.amp);
  fourier_right_eyebrow_Y.sort((a, b) => b.amp - a.amp);
}

function creating_right_eye_outer() {
  const skip = 1;
  for (let i = 0; i < right_eye_outer.length; i += skip) {
    right_eye_outer_x.push(right_eye_outer[i].x * 3);
    right_eye_outer_y.push(right_eye_outer[i].y * 3);
  }
  fourier_right_eye_outer_X = dft(right_eye_outer_x);
  fourier_right_eye_outer_Y = dft(right_eye_outer_y);

  fourier_right_eye_outer_X.sort((a, b) => b.amp - a.amp);
  fourier_right_eye_outer_Y.sort((a, b) => b.amp - a.amp);
}

function creating_right_eye_inner() {
  const skip = 1;
  for (let i = 0; i < right_eye_inner.length; i += skip) {
    right_eye_inner_x.push(right_eye_inner[i].x * 3);
    right_eye_inner_y.push(right_eye_inner[i].y * 3);
  }
  fourier_right_eye_inner_X = dft(right_eye_inner_x);
  fourier_right_eye_inner_Y = dft(right_eye_inner_y);

  fourier_right_eye_inner_X.sort((a, b) => b.amp - a.amp);
  fourier_right_eye_inner_Y.sort((a, b) => b.amp - a.amp);
}

// FACE
function creating_face() {
  const skip = 1;
  for (let i = 0; i < face.length; i += skip) {
    face_x.push(face[i].x * 3);
    face_y.push(face[i].y * 3);
  }
  fourier_face_X = dft(face_x);
  fourier_face_Y = dft(face_y);

  fourier_face_X.sort((a, b) => b.amp - a.amp);
  fourier_face_Y.sort((a, b) => b.amp - a.amp);
}

function setup() {
  createCanvas(1600, 1400);
  creating_lips();
  creating_nose();
  creating_left_eye();
  creating_right_eye();
  creating_left_eyebrow();
  creating_left_eye_outer();
  creating_left_eye_inner();
  creating_right_eyebrow();
  creating_right_eye_outer();
  creating_right_eye_inner();
  creating_face();
}

function epiCycles(x, y, rotation, fourier, element_time) {
  for (let i = 0; i < fourier.length; i++) {
    let prevx = x;
    let prevy = y;
    let freq = fourier[i].freq;
    let radius = fourier[i].amp;
    let phase = fourier[i].phase;
    x += radius * cos(freq * element_time + phase + rotation);
    y += radius * sin(freq * element_time + phase + rotation);

    stroke(255, 100);
    noFill();
    ellipse(prevx, prevy, radius * 2);
    stroke(255);
    line(prevx, prevy, x, y);
  }
  return createVector(x, y);
}

console.log(x.length);

function draw() {
  background(0);
  console.log(lipsPath.length);
  let v_nosex = epiCycles(width / 2 + 100, 100, 0, fouriernoseX, nose_time);
  let v_nosey = epiCycles(
    100,
    height / 2 + 100,
    HALF_PI,
    fouriernoseY,
    nose_time
  );
  let v_nose = createVector(v_nosex.x, v_nosey.y);
  nosePath.unshift(v_nose);
  line(v_nosex.x, v_nosex.y, v_nose.x, v_nose.y);
  line(v_nosey.x, v_nosey.y, v_nose.x, v_nose.y);
  beginShape();
  noFill();
  for (let i = 0; i < nosePath.length; i++) {
    vertex(nosePath[i].x, nosePath[i].y);
  }
  endShape();
  const dt_nose = TWO_PI / fouriernoseY.length;
  nose_time += dt_nose;

  let v_lipsx = epiCycles(width / 2 + 100, 100, 0, fourierlipsX, lips_time);
  let v_lipsy = epiCycles(
    100,
    height / 2 + 100,
    HALF_PI,
    fourierlipsY,
    lips_time
  );
  let v_lips = createVector(v_lipsx.x, v_lipsy.y);
  lipsPath.unshift(v_lips);
  line(v_lipsx.x, v_lipsx.y, v_lips.x, v_lips.y);
  line(v_lipsy.x, v_lipsy.y, v_lips.x, v_lips.y);
  beginShape();
  noFill();
  for (let i = 0; i < lipsPath.length; i++) {
    vertex(lipsPath[i].x, lipsPath[i].y);
  }
  endShape();
  const dt_lips = TWO_PI / fourierlipsY.length;
  lips_time += dt_lips;

  //LEFT PORTION
  let v_left_eye_x = epiCycles(
    width / 2 + 100,
    100,
    0,
    fourier_left_eye_X,
    left_eye_time
  );
  let v_left_eye_y = epiCycles(
    100,
    height / 2 + 100,
    HALF_PI,
    fourier_left_eye_Y,
    left_eye_time
  );
  let v_left_eye = createVector(v_left_eye_x.x, v_left_eye_y.y);
  left_eye_path.unshift(v_left_eye);
  line(v_left_eye_x.x, v_left_eye_x.y, v_left_eye.x, v_left_eye.y);
  line(v_left_eye_y.x, v_left_eye_y.y, v_left_eye.x, v_left_eye.y);
  beginShape();
  noFill();
  for (let i = 0; i < left_eye_path.length; i++) {
    vertex(left_eye_path[i].x, left_eye_path[i].y);
  }
  endShape();
  const dt_left_eye = TWO_PI / fourier_left_eye_Y.length;
  left_eye_time += dt_left_eye;

  let v_left_eyebrow_x = epiCycles(
    width / 2 + 100,
    100,
    0,
    fourier_left_eyebrow_X,
    left_eyebrow_time
  );
  let v_left_eyebrow_y = epiCycles(
    100,
    height / 2 + 100,
    HALF_PI,
    fourier_left_eyebrow_Y,
    left_eyebrow_time
  );
  let v_left_eyebrow = createVector(v_left_eyebrow_x.x, v_left_eyebrow_y.y);
  left_eyebrow_path.unshift(v_left_eyebrow);
  line(
    v_left_eyebrow_x.x,
    v_left_eyebrow_x.y,
    v_left_eyebrow.x,
    v_left_eyebrow.y
  );
  line(
    v_left_eyebrow_y.x,
    v_left_eyebrow_y.y,
    v_left_eyebrow.x,
    v_left_eyebrow.y
  );
  beginShape();
  noFill();
  for (let i = 0; i < left_eyebrow_path.length; i++) {
    vertex(left_eyebrow_path[i].x, left_eyebrow_path[i].y);
  }
  endShape();
  const dt_left_eyebrow = TWO_PI / fourier_left_eyebrow_Y.length;
  left_eyebrow_time += dt_left_eyebrow;

  let v_left_eye_outer_x = epiCycles(
    width / 2 + 100,
    100,
    0,
    fourier_left_eye_outer_X,
    left_eye_outer_time
  );
  let v_left_eye_outer_y = epiCycles(
    100,
    height / 2 + 100,
    HALF_PI,
    fourier_left_eye_outer_Y,
    left_eye_outer_time
  );
  let v_left_eye_outer = createVector(
    v_left_eye_outer_x.x,
    v_left_eye_outer_y.y
  );
  left_eye_outer_path.unshift(v_left_eye_outer);
  line(
    v_left_eye_outer_x.x,
    v_left_eye_outer_x.y,
    v_left_eye_outer.x,
    v_left_eye_outer.y
  );
  line(
    v_left_eye_outer_y.x,
    v_left_eye_outer_y.y,
    v_left_eye_outer.x,
    v_left_eye_outer.y
  );
  beginShape();
  noFill();
  for (let i = 0; i < left_eye_outer_path.length; i++) {
    vertex(left_eye_outer_path[i].x, left_eye_outer_path[i].y);
  }
  endShape();
  const dt_left_eye_outer = TWO_PI / fourier_left_eye_outer_Y.length;
  left_eye_outer_time += dt_left_eye_outer;

  let v_left_eye_inner_x = epiCycles(
    width / 2 + 100,
    100,
    0,
    fourier_left_eye_inner_X,
    left_eye_inner_time
  );
  let v_left_eye_inner_y = epiCycles(
    100,
    height / 2 + 100,
    HALF_PI,
    fourier_left_eye_inner_Y,
    left_eye_inner_time
  );
  let v_left_eye_inner = createVector(
    v_left_eye_inner_x.x,
    v_left_eye_inner_y.y
  );
  left_eye_inner_path.unshift(v_left_eye_inner);
  line(
    v_left_eye_inner_x.x,
    v_left_eye_inner_x.y,
    v_left_eye_inner.x,
    v_left_eye_inner.y
  );
  line(
    v_left_eye_inner_y.x,
    v_left_eye_inner_y.y,
    v_left_eye_inner.x,
    v_left_eye_inner.y
  );
  beginShape();
  noFill();
  for (let i = 0; i < left_eye_inner_path.length; i++) {
    vertex(left_eye_inner_path[i].x, left_eye_inner_path[i].y);
  }
  endShape();
  const dt_left_eye_inner = TWO_PI / fourier_left_eye_inner_Y.length;
  left_eye_inner_time += dt_left_eye_inner;

  // RIGHT PORTION

  let v_right_eye_x = epiCycles(
    width / 2 + 100,
    100,
    0,
    fourier_right_eye_X,
    right_eye_time
  );
  let v_right_eye_y = epiCycles(
    100,
    height / 2 + 100,
    HALF_PI,
    fourier_right_eye_Y,
    right_eye_time
  );
  let v_right_eye = createVector(v_right_eye_x.x, v_right_eye_y.y);
  right_eye_path.unshift(v_right_eye);
  line(v_right_eye_x.x, v_right_eye_x.y, v_right_eye.x, v_right_eye.y);
  line(v_right_eye_y.x, v_right_eye_y.y, v_right_eye.x, v_right_eye.y);
  beginShape();
  noFill();
  for (let i = 0; i < right_eye_path.length; i++) {
    vertex(right_eye_path[i].x, right_eye_path[i].y);
  }
  endShape();
  const dt_right_eye = TWO_PI / fourier_right_eye_Y.length;
  right_eye_time += dt_right_eye;

  let v_right_eyebrow_x = epiCycles(
    width / 2 + 100,
    100,
    0,
    fourier_right_eyebrow_X,
    right_eyebrow_time
  );
  let v_right_eyebrow_y = epiCycles(
    100,
    height / 2 + 100,
    HALF_PI,
    fourier_right_eyebrow_Y,
    right_eyebrow_time
  );
  let v_right_eyebrow = createVector(v_right_eyebrow_x.x, v_right_eyebrow_y.y);
  right_eyebrow_path.unshift(v_right_eyebrow);
  line(
    v_right_eyebrow_x.x,
    v_right_eyebrow_x.y,
    v_right_eyebrow.x,
    v_right_eyebrow.y
  );
  line(
    v_right_eyebrow_y.x,
    v_right_eyebrow_y.y,
    v_right_eyebrow.x,
    v_right_eyebrow.y
  );
  beginShape();
  noFill();
  for (let i = 0; i < right_eyebrow_path.length; i++) {
    vertex(right_eyebrow_path[i].x, right_eyebrow_path[i].y);
  }
  endShape();
  const dt_right_eyebrow = TWO_PI / fourier_right_eyebrow_Y.length;
  right_eyebrow_time += dt_right_eyebrow;

  let v_right_eye_outer_x = epiCycles(
    width / 2 + 100,
    100,
    0,
    fourier_right_eye_outer_X,
    right_eye_outer_time
  );
  let v_right_eye_outer_y = epiCycles(
    100,
    height / 2 + 100,
    HALF_PI,
    fourier_right_eye_outer_Y,
    right_eye_outer_time
  );
  let v_right_eye_outer = createVector(
    v_right_eye_outer_x.x,
    v_right_eye_outer_y.y
  );
  right_eye_outer_path.unshift(v_right_eye_outer);
  line(
    v_right_eye_outer_x.x,
    v_right_eye_outer_x.y,
    v_right_eye_outer.x,
    v_right_eye_outer.y
  );
  line(
    v_right_eye_outer_y.x,
    v_right_eye_outer_y.y,
    v_right_eye_outer.x,
    v_right_eye_outer.y
  );
  beginShape();
  noFill();
  for (let i = 0; i < right_eye_outer_path.length; i++) {
    vertex(right_eye_outer_path[i].x, right_eye_outer_path[i].y);
  }
  endShape();
  const dt_right_eye_outer = TWO_PI / fourier_right_eye_outer_Y.length;
  right_eye_outer_time += dt_right_eye_outer;

  let v_right_eye_inner_x = epiCycles(
    width / 2 + 100,
    100,
    0,
    fourier_right_eye_inner_X,
    right_eye_inner_time
  );
  let v_right_eye_inner_y = epiCycles(
    100,
    height / 2 + 100,
    HALF_PI,
    fourier_right_eye_inner_Y,
    right_eye_inner_time
  );
  let v_right_eye_inner = createVector(
    v_right_eye_inner_x.x,
    v_right_eye_inner_y.y
  );
  right_eye_inner_path.unshift(v_right_eye_inner);
  line(
    v_right_eye_inner_x.x,
    v_right_eye_inner_x.y,
    v_right_eye_inner.x,
    v_right_eye_inner.y
  );
  line(
    v_right_eye_inner_y.x,
    v_right_eye_inner_y.y,
    v_right_eye_inner.x,
    v_right_eye_inner.y
  );
  beginShape();
  noFill();
  for (let i = 0; i < right_eye_inner_path.length; i++) {
    vertex(right_eye_inner_path[i].x, right_eye_inner_path[i].y);
  }
  endShape();
  const dt_right_eye_inner = TWO_PI / fourier_right_eye_inner_Y.length;
  right_eye_inner_time += dt_right_eye_inner;

  // FACE
  let v_face_x = epiCycles(width / 2 + 100, 100, 0, fourier_face_X, face_time);
  let v_face_y = epiCycles(
    100,
    height / 2 + 100,
    HALF_PI,
    fourier_face_Y,
    face_time
  );
  let v_face = createVector(v_face_x.x, v_face_y.y);
  face_path.unshift(v_face);
  line(v_face_x.x, v_face_x.y, v_face.x, v_face.y);
  line(v_face_y.x, v_face_y.y, v_face.x, v_face.y);
  beginShape();
  noFill();
  for (let i = 0; i < face_path.length; i++) {
    vertex(face_path[i].x, face_path[i].y);
  }
  endShape();
  const dt_face = TWO_PI / fourier_face_Y.length;
  face_time += dt_face;

  if (time > TWO_PI) {
    time = 0;
    path = [];
    lipsPath = [];
    nosePath = [];
  }
}
