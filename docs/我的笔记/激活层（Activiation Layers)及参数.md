�ڼ�����У����������ݽ��м��������ʵ���Ͼ���һ�ֺ����任��������Ԫ�ؽ�������ġ���bottom�õ�һ��blob�������룬����󣬴�top����һ��blob���ݡ�����������У�û�иı����ݵĴ�С�����������������ݴ�С����ȵġ�

���룺n*c*h*w

�����n*c*h*w

���õļ������sigmoid, tanh,relu�ȣ�����ֱ���ܡ�

1��Sigmoid

��ÿ���������ݣ�����sigmoid����ִ�в��������ֲ����ñȽϼ򵥣�û�ж���Ĳ�����



�����ͣ�Sigmoid

ʾ����

layer {
  name: "encode1neuron"
  bottom: "encode1"
  top: "encode1neuron"
  type: "Sigmoid"
}
2��ReLU / Rectified-Linear and Leaky-ReLU

ReLU��Ŀǰʹ�����ļ��������Ҫ��Ϊ���������죬�����ܱ���ͬ��Ч����

��׼��ReLU����Ϊmax(x, 0)����x>0ʱ�����x; ��x<=0ʱ�����0

f(x)=max(x,0)

�����ͣ�ReLU

��ѡ������

����negative_slope��Ĭ��Ϊ0. �Ա�׼��ReLU�������б仯��������������ֵ����ô����Ϊ����ʱ���Ͳ�������Ϊ0��������ԭʼ���ݳ���negative_slope

layer {
  name: "relu1"
  type: "ReLU"
  bottom: "pool1"
  top: "pool1"
}
RELU��֧��in-place���㣬����ζ��bottom�������������ͬ�Ա����ڴ�����ġ�

3��TanH / Hyperbolic Tangent

����˫�����к��������ݽ��б任��



�����ͣ�TanH

layer {
  name: "layer"
  bottom: "in"
  top: "out"
  type: "TanH"
}
4��Absolute Value

��ÿ���������ݵľ���ֵ��

f(x)=Abs(x)

�����ͣ�AbsVal

layer {
  name: "layer"
  bottom: "in"
  top: "out"
  type: "AbsVal"
}
5��Power

��ÿ���������ݽ���������

f(x)= (shift + scale * x) ^ power

�����ͣ�Power

��ѡ������

����power: Ĭ��Ϊ1

����scale: Ĭ��Ϊ1

����shift: Ĭ��Ϊ0

���ƴ���
layer {
  name: "layer"
  bottom: "in"
  top: "out"
  type: "Power"
  power_param {
    power: 2
    scale: 1
    shift: 0
  }
}
���ƴ���
6��BNLL

binomial normal log likelihood�ļ��

f(x)=log(1 + exp(x))

�����ͣ�BNLL

layer {
  name: "layer"
  bottom: "in"
  top: "out"
  type: ��BNLL��
}