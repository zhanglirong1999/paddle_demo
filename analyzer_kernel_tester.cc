#include <fstream>
#include <iostream>
#include <gtest/gtest.h>
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/fluid/platform/enforce.h"
#include <paddle/fluid/framework/op_registry.h>
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/framework/type_defs.h"

namespace paddle {
namespace inference {
namespace analysis {

TEST(Analyzer_kernel, fused_matmul) {
    std::string type_ = "fusion_gru";
    auto& all_op_kernels = framework::OperatorWithKernel::AllOpKernels();
    for (const auto& kernel_pair : all_op_kernels) {
        std::cout << kernel_pair.first << std::endl;
    }
    auto kernels_iter = all_op_kernels.find(type_);
    std::cout << "1111111" << std::endl;
    std::cout << "kernels_iter: " << kernels_iter->first << std::endl;
    std::cout << "all_op_kernels.end(): " << all_op_kernels.end()->first << std::endl;
    PADDLE_ENFORCE_NE(
      kernels_iter,
      all_op_kernels.end(),
      platform::errors::Unimplemented(
          "There are no kernels which are registered in the %s operator.",
          type_));
}

TEST(Analyzer_kernel,run_phi ) {
    std::string type_ = "fused_matmul";
    // std::string type_ = "fusion_gru";
    if (phi::KernelFactory::Instance().HasCompatiblePhiKernel(type_)) {
        std::cout << "1111111" << std::endl;
    }else{
        std::cout << "2222222222" << std::endl;
    }

    auto& pool = platform::DeviceContextPool::Instance();
    auto* dev_ctx = pool.Get(platform::CPUPlace());

    paddle::framework::Scope scope;
    // runtime_scope->Var("X")->GetMutable<phi::DenseTensor>();
    // runtime_scope->Var("Y")->GetMutable<phi::DenseTensor>();
    // runtime_scope->Var("Out")->GetMutable<phi::DenseTensor>();
    auto var_x = scope.Var("X");
    auto tensor_x = var_x->GetMutable<phi::DenseTensor>();
    tensor_x->Resize({1, 2}); 
    auto pr1 = tensor_x->mutable_data<float>(platform::CPUPlace());
    pr1[0] = 1.0;
    pr1[1] = 2.0;

    auto* var_y = scope.Var("Y");
    auto tensor_y = var_y->GetMutable<phi::DenseTensor>();
    tensor_y->Resize({1, 2}); 
    auto pr2 = tensor_y->mutable_data<float>(platform::CPUPlace());
    pr2[0] = 1.0;
    pr2[1] = 2.0;

    auto* var_out = scope.Var("Out");
    auto tensor_out = var_out->GetMutable<phi::DenseTensor>();
    tensor_out->Resize({1, 2}); 
    auto pr3 = tensor_out ->mutable_data<float>(platform::CPUPlace());
    pr3[0] = 1.0;
    pr3[1] = 2.0;


    framework::AttributeMap attrs;
    attrs.insert({"use_mkldnn", {true}});

    // auto op = framework::OpRegistry::CreateOp(type_,
    // {{"X",{{1, 2}, {3, 4}}}, {"Y", {{5, 6}, {7, 8}}}},
    // {{"Out",{{0, 0}, {0, 0}}}},
    // attrs);
    auto op = framework::OpRegistry::CreateOp(type_,
                                            {{"X", {"X"}},
                                            {"Y", {"Y"}}},
                                            {{"Out", {"Out"}}},
                                            attrs);
                                            
    if (dynamic_cast<framework::OperatorWithKernel*>(op.get()) == nullptr){
    //std::cout << op_with_kernel->Type() << std::endl;
        std::cout << "come in this" << std::endl;
    }else{
        auto op_with_kernel = const_cast<framework::OperatorWithKernel*>(
            static_cast<const framework::OperatorWithKernel*>(op.get()));
        std::cout << op_with_kernel->Type() << std::endl;

        paddle::framework::RuntimeContext runtime_context({},{});
        runtime_context.inputs["X"] = {var_x};
        runtime_context.inputs["Y"] = {var_y};
        runtime_context.outputs["Out"] = {var_out};

        auto exec_ctx = paddle::framework::ExecutionContext(
            *op_with_kernel, scope, *dev_ctx, runtime_context);
        auto expected_kernel_key = framework::TransPhiKernelKeyToOpKernelType(
            op_with_kernel->GetExpectedKernelType(exec_ctx));
        std::cout << "expected_kernel_key : " << expected_kernel_key << std::endl;
        if (phi::KernelFactory::Instance().HasCompatiblePhiKernel(
                op_with_kernel->Type())) {
          auto phi_kernel_key = op_with_kernel->ChoosePhiKernel(exec_ctx);
          auto phi_kernel_name = op_with_kernel->PhiKernelSignature()->name;
          std::cout << "phi_kernel_key:" << phi_kernel_key << std::endl;
          std::cout << "phi_kernel_name:" << phi_kernel_name << std::endl;
        //   std::cout << phi_kernel_name << std::endl;
          bool in_custom_back_list = false;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
          in_custom_back_list =
              phi::backends::custom_device::is_in_custom_black_list(
                  phi_kernel_name);
#endif
          if (op_with_kernel->PhiKernel()->IsValid() && !in_custom_back_list){
            std::cout << "phi kernel can run!" << std::endl;
          }else{
            if (!op_with_kernel->SupportsKernelType(expected_kernel_key,
                                                     exec_ctx)){
                std::cout << "can come in expected_kernel_key" << std::endl;
                auto phi_cpu_kernel_key =
                  FallBackToCpu(phi_kernel_key, *op_with_kernel);
                op_with_kernel->ResetPhiKernel(
                  new phi::Kernel(phi::KernelFactory::Instance().SelectKernel(
                      phi_kernel_name, phi_cpu_kernel_key)));
                if (op_with_kernel->PhiKernel()->IsValid()) {
                    std::cout << "phi can run after reset" << std::endl;
                }else{
                    std::cout << "still can not run!" << std::endl;
                }
                }
          }
        }

    }
}

TEST(Analyzer_kernel,new_run_phi ){
    std::string type_ = "fused_matmul";
    // std::string type_ = "fusion_gru";
    if (phi::KernelFactory::Instance().HasCompatiblePhiKernel(type_)) {
        std::cout << "1111111" << std::endl;
    }else{
        std::cout << "2222222222" << std::endl;
    }
    framework::AttributeMap attrs;
    attrs.insert({"use_mkldnn", {true}});

    auto op = framework::OpRegistry::CreateOp(type_,
    {{"X",{{1, 2, 3, 4}}}, {"Y", {{5, 6, 7, 8}}}},
    {{"Out",{{0, 0, 0, 0}}}},
    attrs);

    auto op_with_kernel = const_cast<framework::OperatorWithKernel*>(
            static_cast<const framework::OperatorWithKernel*>(op.get()));
    // if (op_with_kernel->PhiKernel()->IsValid()){
    //         std::cout << "333333333333333333333333333" << std::endl;
    // }

}


}  // namespace analysis
}  // namespace inference
}  // namespace paddle