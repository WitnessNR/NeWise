import numpy as np
from cnn_bounds_full_core_with_LP import run_certified_bounds_core, run_verified_robustness_core
from cnn_bounds_full_with_LP import run_certified_bounds, run_verified_robustness

import time as timing
import datetime

if __name__ == '__main__':
    
    ## Table 1
    # # models_with_nonnegative_weights
    
    # models with sigmoid
    path_prefix = "models/models_with_positive_weights/sigmoid/"
    
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_9118.h5', 100, 105, 1, method='NeWise', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_9118.h5', 100, 105, 1, method='DeepCert', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_9118.h5', 100, 105, 1, method='VeriNet', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_9118.h5', 100, 105, 1, method='RobustVerifier', mnist=True)
    
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_9162.h5', 100, 105, 1, method='NeWise', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_9162.h5', 100, 105, 1, method='DeepCert', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_9162.h5', 100, 105, 1, method='VeriNet', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_9162.h5', 100, 105, 1, method='RobustVerifier', mnist=True)
    
    # run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_8858.h5', 100, 105, 1, method='NeWise', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_8858.h5', 100, 105, 1, method='DeepCert', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_8858.h5', 100, 105, 1, method='VeriNet', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_8858.h5', 100, 105, 1, method='RobustVerifier', mnist=True)
    
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_9420.h5', 100, 105, 1, method='NeWise', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_9420.h5', 100, 105, 1, method='DeepCert', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_9420.h5', 100, 105, 1, method='VeriNet', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_9420.h5', 100, 105, 1, method='RobustVerifier', mnist=True)
    
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_9630.h5', 100, 105, 1, method='NeWise', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_9630.h5', 100, 105, 1, method='DeepCert', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_9630.h5', 100, 105, 1, method='VeriNet', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_9630.h5', 100, 105, 1, method='RobustVerifier', mnist=True)
    
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_9537.h5', 100, 105, 1, method='NeWise', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_9537.h5', 100, 105, 1, method='DeepCert', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_9537.h5', 100, 105, 1, method='VeriNet', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_9537.h5', 100, 105, 1, method='RobustVerifier', mnist=True)
    
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_9120.h5', 100, 105, 1, method='NeWise', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_9120.h5', 100, 105, 1, method='DeepCert', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_9120.h5', 100, 105, 1, method='VeriNet', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_9120.h5', 100, 105, 1, method='RobustVerifier', mnist=True)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_9151.h5', 100, 105, 1, method='NeWise', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_9151.h5', 100, 105, 1, method='DeepCert', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_9151.h5', 100, 105, 1, method='VeriNet', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_9151.h5', 100, 105, 1, method='RobustVerifier', mnist=True)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_8563.h5', 100, 105, 1, method='NeWise', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_8563.h5', 100, 105, 1, method='DeepCert', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_8563.h5', 100, 105, 1, method='VeriNet', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_8563.h5', 100, 105, 1, method='RobustVerifier', mnist=True)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights.h5', 100, 105, 1, method='NeWise', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights.h5', 100, 105, 1, method='DeepCert', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights.h5', 100, 105, 1, method='VeriNet', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights.h5', 100, 105, 1, method='RobustVerifier', mnist=True)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights.h5', 100, 105, 1, method='NeWise', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights.h5', 100, 105, 1, method='DeepCert', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights.h5', 100, 105, 1, method='VeriNet', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights.h5', 100, 105, 1, method='RobustVerifier', mnist=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_4056_cpu.h5', 100, 105, 1, method='NeWise', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_4056_cpu.h5', 100, 105, 1, method='DeepCert', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_4056_cpu.h5', 100, 105, 1, method='VeriNet', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_4056_cpu.h5', 100, 105, 1, method='RobustVerifier', cifar=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_4298_cpu.h5', 100, 105, 1, method='NeWise', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_4298_cpu.h5', 100, 105, 1, method='DeepCert', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_4298_cpu.h5', 100, 105, 1, method='VeriNet', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_4298_cpu.h5', 100, 105, 1, method='RobustVerifier', cifar=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_4422_cpu.h5', 100, 105, 1, method='NeWise', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_4422_cpu.h5', 100, 105, 1, method='DeepCert', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_4422_cpu.h5', 100, 105, 1, method='VeriNet', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_4422_cpu.h5', 100, 105, 1, method='RobustVerifier', cifar=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_3988_cpu.h5', 100, 105, 1, method='NeWise', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_3988_cpu.h5', 100, 105, 1, method='DeepCert', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_3988_cpu.h5', 100, 105, 1, method='VeriNet', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_3988_cpu.h5', 100, 105, 1, method='RobustVerifier', cifar=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_4033_cpu.h5', 100, 105, 1, method='NeWise', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_4033_cpu.h5', 100, 105, 1, method='DeepCert', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_4033_cpu.h5', 100, 105, 1, method='VeriNet', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_4033_cpu.h5', 100, 105, 1, method='RobustVerifier', cifar=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_4799_cpu.h5', 100, 105, 1, method='NeWise', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_4799_cpu.h5', 100, 105, 1, method='DeepCert', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_4799_cpu.h5', 100, 105, 1, method='VeriNet', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_4799_cpu.h5', 100, 105, 1, method='RobustVerifier', cifar=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_4521_cpu.h5', 100, 105, 1, method='NeWise', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_4521_cpu.h5', 100, 105, 1, method='DeepCert', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_4521_cpu.h5', 100, 105, 1, method='VeriNet', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_4521_cpu.h5', 100, 105, 1, method='RobustVerifier', cifar=True)
    
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_3812.h5', 100, 105, 1, method='NeWise', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_3812.h5', 100, 105, 1, method='DeepCert', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_3812.h5', 100, 105, 1, method='VeriNet', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_3812.h5', 100, 105, 1, method='RobustVerifier', cifar=True)
    
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_3597.h5', 100, 105, 1, method='NeWise', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_3597.h5', 100, 105, 1, method='DeepCert', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_3597.h5', 100, 105, 1, method='VeriNet', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_3597.h5', 100, 105, 1, method='RobustVerifier', cifar=True)
    
    # models with tanh
    path_prefix = "models/models_with_positive_weights/tanh/"
    
    # trained on MNIST
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_tanh_9546.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_tanh_9546.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_tanh_9546.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_tanh_9546.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True)
    
    
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_tanh_9644.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_tanh_9644.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_tanh_9644.h5', 100, 105, 1, method='VeriNet',activation='tanh', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_tanh_9644.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True)
    
    # run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_tanh_9573.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_tanh_9573.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_tanh_9573.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_tanh_9573.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True)
    
    
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_tanh_9637.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_tanh_9637.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_tanh_9637.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_tanh_9637.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True)
    
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_tanh_9648.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_tanh_9648.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_tanh_9648.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_tanh_9648.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True)
    
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_tanh_9624.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_tanh_9624.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_tanh_9624.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_tanh_9624.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True)
    
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_tanh_9530.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_tanh_9530.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_tanh_9530.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_tanh_9530.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_tanh_9612.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_tanh_9612.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_tanh_9612.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_tanh_9612.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_tanh_9655.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_tanh_9655.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_tanh_9655.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_tanh_9655.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights_tanh_9457.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights_tanh_9457.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights_tanh_9457.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights_tanh_9457.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights_tanh_9449.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights_tanh_9449.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights_tanh_9449.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights_tanh_9449.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True)
    
    # # trained on CIFAR10
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_tanh_38.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_tanh_38.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_tanh_38.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_tanh_38.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_tanh_3281.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_tanh_3281.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_tanh_3281.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_tanh_3281.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_tanh_3581.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_tanh_3581.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_tanh_3581.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_tanh_3581.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_tanh_3114.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_tanh_3114.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_tanh_3114.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_tanh_3114.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_tanh_2823.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_tanh_2823.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_tanh_2823.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_tanh_2823.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True)
    
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_tanh_2988.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_tanh_2988.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_tanh_2988.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_tanh_2988.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_tanh_2906.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_tanh_2906.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_tanh_2906.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_tanh_2906.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True)
    
    
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_tanh_3853.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_tanh_3853.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_tanh_3853.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_tanh_3853.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True)
    
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_tanh_4031.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_tanh_4031.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_tanh_4031.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_tanh_4031.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True)
    
    
    # Table 2
    # models with mixed weights
    
    path_prefix = "models/converted_model/"
    
    # VNN-COMP 2021 benchmark
    # run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6x200.h5', 100, 105, 1, method='NeWise', eran_fnn=True, mnist=True)
    # run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6x200.h5', 100, 105, 1, method='DeepCert', eran_fnn=True, mnist=True)
    # run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6x200.h5', 100, 105, 1, method='VeriNet', eran_fnn=True, mnist=True)
    # run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6x200.h5', 100, 105, 1, method='RobustVerifier', eran_fnn=True, mnist=True)
    
    
    # DeepPoly benchmark
    # run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6_500.h5', 100, 105, 1, method='NeWise', eran_fnn=True, mnist=True)
    # run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6_500.h5', 100, 105, 1, method='DeepCert', eran_fnn=True, mnist=True)
    # run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6_500.h5', 100, 105, 1, method='VeriNet', eran_fnn=True, mnist=True)
    # run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6_500.h5', 100, 105, 1, method='RobustVerifier', eran_fnn=True, mnist=True)
    
    # run_certified_bounds(path_prefix + 'ffnnSIGMOID__PGDK_w_0.1_6_500.h5', 100, 105, 1, method='NeWise', eran_fnn=True, mnist=True)
    # run_certified_bounds(path_prefix + 'ffnnSIGMOID__PGDK_w_0.1_6_500.h5', 100, 105, 1, method='DeepCert', eran_fnn=True, mnist=True)
    # run_certified_bounds(path_prefix + 'ffnnSIGMOID__PGDK_w_0.1_6_500.h5', 100, 105, 1, method='VeriNet', eran_fnn=True, mnist=True)
    # run_certified_bounds(path_prefix + 'ffnnSIGMOID__PGDK_w_0.1_6_500.h5', 100, 105, 1, method='RobustVerifier', eran_fnn=True, mnist=True)
    
    # run_certified_bounds(path_prefix + 'ffnnSIGMOID__PGDK_w_0.3_6_500.h5', 100, 105, 1, method='NeWise', eran_fnn=True, mnist=True)
    # run_certified_bounds(path_prefix + 'ffnnSIGMOID__PGDK_w_0.3_6_500.h5', 100, 105, 1, method='DeepCert', eran_fnn=True, mnist=True)
    # run_certified_bounds(path_prefix + 'ffnnSIGMOID__PGDK_w_0.3_6_500.h5', 100, 105, 1, method='VeriNet', eran_fnn=True, mnist=True)
    # run_certified_bounds(path_prefix + 'ffnnSIGMOID__PGDK_w_0.3_6_500.h5', 100, 105, 1, method='RobustVerifier', eran_fnn=True, mnist=True)
    
    # run_certified_bounds(path_prefix + 'convMedGSIGMOID__Point.h5', 100, 105, 1, method='NeWise', eran_cnn=True, mnist=True)
    # run_certified_bounds(path_prefix + 'convMedGSIGMOID__Point.h5', 100, 105, 1, method='DeepCert', eran_cnn=True, mnist=True)
    # run_certified_bounds(path_prefix + 'convMedGSIGMOID__Point.h5', 100, 105, 1, method='VeriNet', eran_cnn=True, mnist=True)
    # run_certified_bounds(path_prefix + 'convMedGSIGMOID__Point.h5', 100, 105, 1, method='RobustVerifier', eran_cnn=True, mnist=True)


    # CNN-Cert benchmark 
    
    path_prefix = "models/public_benchmarks/"
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_sigmoid', 100, 105, 1, method='NeWise', cnn_cert_model=True, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_sigmoid', 100, 105, 1, method='DeepCert', cnn_cert_model=True, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_sigmoid', 100, 105, 1, method='VeriNet', cnn_cert_model=True, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_sigmoid', 100, 105, 1, method='RobustVerifier', cnn_cert_model=True, mnist=True)
    
    path_prefix = "models/before_models/"
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_sigmoid_myself_9003.h5', 100, 105, 1, method='NeWise', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_sigmoid_myself_9003.h5', 100, 105, 1, method='DeepCert', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_sigmoid_myself_9003.h5', 100, 105, 1, method='VeriNet', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_sigmoid_myself_9003.h5', 100, 105, 1, method='RobustVerifier', mnist=True)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_sigmoid_myself_8891.h5', 100, 105, 1, method='NeWise', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_sigmoid_myself_8891.h5', 100, 105, 1, method='DeepCert', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_sigmoid_myself_8891.h5', 100, 105, 1, method='VeriNet', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_sigmoid_myself_8891.h5', 100, 105, 1, method='RobustVerifier', mnist=True)
    
    path_prefix = "models/public_benchmarks/"
    run_certified_bounds_core(path_prefix + 'mnist_cnn_8layer_5_3_sigmoid', 100, 105, 1, method='NeWise', cnn_cert_model=True, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_8layer_5_3_sigmoid', 100, 105, 1, method='DeepCert', cnn_cert_model=True, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_8layer_5_3_sigmoid', 100, 105, 1, method='VeriNet', cnn_cert_model=True, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_8layer_5_3_sigmoid', 100, 105, 1, method='RobustVerifier', cnn_cert_model=True, mnist=True)
    
    path_prefix = "models/before_models/"
    run_certified_bounds_core(path_prefix + 'mnist_cnn_10layer_5_3_sigmoid_myself_5873.h5', 100, 105, 1, method='NeWise', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_10layer_5_3_sigmoid_myself_5873.h5', 100, 105, 1, method='DeepCert', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_10layer_5_3_sigmoid_myself_5873.h5', 100, 105, 1, method='VeriNet', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_10layer_5_3_sigmoid_myself_5873.h5', 100, 105, 1, method='RobustVerifier', mnist=True)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='NeWise', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True)

    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3_sigmoid_myself_7379.h5', 100, 105, 1, method='NeWise', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3_sigmoid_myself_7379.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3_sigmoid_myself_7379.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3_sigmoid_myself_7379.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True)
    
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='NeWise', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='NeWise', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True)
    
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='NeWise', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True)
    
    path_prefix = "models/public_benchmarks/"
    run_certified_bounds_core(path_prefix + 'cifar_cnn_5layer_5_3_sigmoid', 100, 105, 1, method='NeWise', cnn_cert_model=True, cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar_cnn_5layer_5_3_sigmoid', 100, 105, 1, method='DeepCert', cnn_cert_model=True, cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar_cnn_5layer_5_3_sigmoid', 100, 105, 1, method='VeriNet', cnn_cert_model=True, cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar_cnn_5layer_5_3_sigmoid', 100, 105, 1, method='RobustVerifier', cnn_cert_model=True, cifar=True)
    
    
    ## Appendix Table 2
    # Atan models with nonnegative weights
    
    path_prefix = "models/models_with_positive_weights/arctan/"
    
    # trained on MNIST
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_atan_9327.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_atan_9327.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_atan_9327.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_atan_9327.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True)
    
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_atan_9662.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_atan_9662.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_atan_9662.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_atan_9662.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True)
    
    # run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_atan_9233.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_atan_9233.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_atan_9233.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_atan_9233.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True)
    
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_atan_9700.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_atan_9700.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_atan_9700.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_atan_9700.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True)
    
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_atan_9706.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_atan_9706.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_atan_9706.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_atan_9706.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True)
    
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_atan_9650.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_atan_9650.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_atan_9650.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True)
    # run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_atan_9650.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True)
    
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_atan_9499.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_atan_9499.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_atan_9499.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_atan_9499.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_atan_9481.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_atan_9481.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_atan_9481.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_atan_9481.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_atan_9678.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_atan_9678.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_atan_9678.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_atan_9678.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights_atan_9575.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights_atan_9575.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights_atan_9575.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights_atan_9575.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights_atan_9389.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights_atan_9389.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights_atan_9389.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights_atan_9389.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True)

    # # trained on CIFAR10
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_atan_3875.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_atan_3875.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_atan_3875.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_atan_3875.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_atan_4078.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_atan_4078.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_atan_4078.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_atan_4078.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_atan_4291.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_atan_4291.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_atan_4291.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_atan_4291.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_6x100_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_6x100_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_6x100_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_6x100_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True)
    
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True)
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True)
   
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_atan_3782.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_atan_3782.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_atan_3782.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_atan_3782.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True)
    
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True)
    
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True)
