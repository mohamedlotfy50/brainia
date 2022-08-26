// import 'dart:math' as math;

// import 'package:dart_ml/src/tensor/core/tensor.dart';

// Tensor log(dynamic input) {
//   var t1 = TensorHelper.toTensor(input);

//   var output = <num>[];

//   for (var i = 0; i < t1.size; i++) {
//     var t1Val = t1.getAt([i]);
//     output.add(math.log(t1Val));
//   }
//   var tensor = Tensor(output);
//   if (t1.shape.isNotEmpty) {
//     tensor.reshape(t1.shape);
//   }

//   return tensor;
// }
