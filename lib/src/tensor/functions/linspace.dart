part of '../core/tensor.dart';

Tensor<T> linspace<T extends num>(num start, num end,
    {int number = 50, endpoint = true}) {
  var output = [];
  num delta;
  if (endpoint) {
    delta = (end - start) / (number - 1);
  } else {
    delta = (end - start) / number;
  }

  for (var i = 0; i < number; i++) {
    output.add(start + i * delta);
  }
  return Tensor<T>(output);
}
