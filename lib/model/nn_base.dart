import 'package:dtensor/dtensor.dart';

abstract class Neural {
  DTensor<num> forward(DTensor<num> input);

  DTensor<num> backward(DTensor<num> input) {
    throw Exception();
  }

  DTensor<num> call(DTensor<num> input) {
    return forward(input);
  }
}
