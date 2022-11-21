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

#include "onnx_op_converter.h"
#include "onnx_utility.h"

DECLARE_OP_CONVERTER(GridSample);

string OnnxOpConverterGridSample::TNNOpType(NodeProto &node,
                                           OnnxNetInfo &net_info) {
    return "GridSample";
}

string OnnxOpConverterGridSample::TNNLayerParam(NodeProto &node,
                                               OnnxNetInfo &net_info) {
    ostringstream layer_param;
    auto mode = get_node_attr_s(node, "mode", "bilinear");
    if (mode.size() > 0 && mode == "bilinear") {
        //bilinear
        layer_param << "2 ";
    } else {
        LOGE("GridSample dont support mode\n");
        return "";
    }

    auto pade_type = get_node_attr_s(node, "padding_mode", "zeros");
    if (pade_type.size() > 0 && pade_type == "zeros") {
        //padding zeros
        layer_param << "0 ";
    } else {
        LOGE("GridSample dont support pade_type\n");
        return "";
    }

    auto align_corners = get_node_attr_i(node, "align_corners", 0);
    if (0 == align_corners || 1 == align_corners) {
        layer_param << (0 == align_corners ? "0 " : "1 ");
    } else {
        LOGE("GridSample dont support align_corners\n");
        return "";
    }

    return layer_param.str();
}

bool OnnxOpConverterGridSample::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterGridSample::WriteTNNModel(Serializer *net_writer,
                                            NodeProto &node,
                                            OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(GridSample, GridSample);
