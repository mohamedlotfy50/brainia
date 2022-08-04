import 'dart:io' show stdout;

void start() {
  print('building code');

  var size = 99999;

  for (var i = 0; i <= size; i++) {
    drawProgressBar(i, size);
  }
}

void drawProgressBar(num amount, num size) {
  var terminalSize = stdout.terminalColumns ~/ 2 - '$size'.length;
  var precentage = amount / size;
  var bars = terminalSize * precentage;
  final color = barColor(precentage);
  stdout.write(color +
      '|' +
      '=' * (bars.toInt() - 1) +
      '>' +
      '-' * (terminalSize - bars.toInt() - 1) +
      '|' +
      '\x1b[0m' +
      ' $amount/$size');
}

String barColor(double val) {
  if (val < 1 / 3) {
    return '\r\x1B[31m';
  } else if (val > 1 / 3 && val < 2 / 3) {
    return '\r\x1B[33m';
  } else {
    return '\r\x1b[34m';
  }
}
