let x = [];
let y = [];
let fourierX;
let fourierY;
let time = 0;
let path = [];

function setup() {
  createCanvas(1600, 800);
  const skip = 5;
  for (let i = 0; i < ramanujan.length; i += skip) {
    x.push(ramanujan[i].x);
    y.push(ramanujan[i].y);
  }
  fourierX = dft(x);
  fourierY = dft(y);

  fourierX.sort((a, b) => b.amp - a.amp);
  fourierY.sort((a, b) => b.amp - a.amp);
}

function epiCycles(x, y, rotation, fourier) {
  for (let i = 0; i < fourier.length; i++) {
    let prevx = x;
    let prevy = y;
    let freq = fourier[i].freq;
    let radius = fourier[i].amp;
    let phase = fourier[i].phase;
    x += radius * cos(freq * time + phase + rotation);
    y += radius * sin(freq * time + phase + rotation);

    stroke(255, 100);
    noFill();
    ellipse(prevx, prevy, radius * 2);
    stroke(255);
    line(prevx, prevy, x, y);
  }
  return createVector(x, y);
}

function draw() {
  background(0);

  let vx = epiCycles(width / 2 + 100, 100, 0, fourierX);
  let vy = epiCycles(100, height / 2 + 100, HALF_PI, fourierY);
  let v = createVector(vx.x, vy.y);
  path.unshift(v);

  line(vx.x, vx.y, v.x, v.y);
  line(vy.x, vy.y, v.x, v.y);

  beginShape();
  noFill();
  for (let i = 0; i < path.length; i++) {
    vertex(path[i].x, path[i].y);
  }
  endShape();

  const dt = TWO_PI / fourierY.length;
  time += dt;
}
