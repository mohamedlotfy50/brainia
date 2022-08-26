// import 'package:dart_ml/src/tensor/core/tensor.dart';

// Tensor<T> maximum<T extends num>(dynamic input1, dynamic input2) {
//   var t1 = TensorHelper.toTensor<T>(input1),
//       t2 = TensorHelper.toTensor<T>(input2);

//   if (t2.size > t1.size) {
//     var temp = t1;
//     t1 = t2;
//     t2 = temp;
//   }
//   var output = <num>[];
//   var broadcast = TensorHelper.isBroadcastable(t1.shape, t2.shape);
//   if (broadcast != null) {
//     for (var i = 0; i < t1.size; i++) {
//       var t1Val = t1.getAt([i]);
//       var t2Val = t2.getAt([i % t2.size]);
//       if (t1Val > t2Val) {
//         output.add(t1Val);
//       } else {
//         output.add(t2Val);
//       }
//     }

//     return Tensor(output)..reshape(t1.shape);
//   } else {
//     throw Exception('not prodcastable');
//   }
// }
