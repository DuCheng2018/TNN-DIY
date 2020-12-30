// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "graph/attr_value.h"
#include "npu_base_layer_convert.h"
#include "npu_utils.h"


namespace TNN_NS {

DECLARE_NPU_LAYER_WEIGHT(Upsample, LAYER_UPSAMPLE)

Status NpuUpsampleLayer::Convert() {

    auto param = dynamic_cast<UpsampleLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    float scale_w = param->scales[0];
    float scale_h = param->scales[1];

    if (param->dims.size() >= 2) {
        scale_w = param->dims[0] / input_ops_[0]->GetShape()[3];
        scale_h = param->dims[1] / input_ops_[0]->GetShape()[2];
    }

    const int resize_mode = param->mode;
    const bool align_corners = param->align_corners;
    std::vector<int> dims_vec = param->dims;

    if (resize_mode==2 && !dims_vec.empty()) {
        const int height = dims_vec[1];
        const int width = dims_vec[0];
        // target size
        std::shared_ptr<ge::op::Const> input_size_const = std::make_shared<ge::op::Const>(layer_name_ + "_input_size");
        ge::TensorDesc desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
        NpuUtils::CreateAttrArray(input_size_const, std::vector<int> {height, width}, desc, 2);
        weight_ops_.push_back(input_size_const);
        auto output = std::make_shared<hiai::op::ResizeBilinearV2>(outputs_name_[0]);
        output->set_input_x(*input_ops_[0]->GetOperator());
        output->set_input_size(*input_size_const);
        output->set_attr_align_corners(align_corners);
        ADD_OUTPUT_OP(output)
    } else {
        // only support scale is int
        if (scale_w != (int)scale_w || scale_h != (int)scale_h) {
            LOGE("the upsample scale is not support in huawei NPU\n");
            return Status(TNNERR_NPU_UNSUPPORT_ERROR, "the upsample scale is not support in huawei NPU");
        }
        auto output = std::make_shared<hiai::op::Upsample>(outputs_name_[0]);
        output->set_input_x(*input_ops_[0]->GetOperator());
        output->set_attr_stride_h((int)scale_h);
        output->set_attr_stride_w((int)scale_w);
        ADD_OUTPUT_OP(output)
    }
}

REGISTER_NPU_LAYER(Upsample, LAYER_UPSAMPLE)

} // namespace TNN_NS
