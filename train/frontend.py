import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class ImageFeatureExtractor(nn.Module):
    """
    图像特征提取模块 - "卷积残差池化一条龙"
    输入: RGB图像 [batch_size, 3, height, width]
    输出: 特征图 [batch_size, feature_dim, feat_height, feat_width]
    """

    def __init__(self, backbone_name='resnet18', pretrained=True, feature_dim=256):
        super(ImageFeatureExtractor, self).__init__()

        # 加载预训练的ResNet骨干网络
        backbone = models.__dict__[backbone_name](pretrained=pretrained)

        # 移除最后的全连接层和池化层
        self.features = nn.Sequential(*list(backbone.children())[:-2])

        # 获取骨干网络的输出通道数
        in_channels = backbone.fc.in_features if hasattr(backbone, 'fc') else 512

        # 添加1x1卷积调整特征维度
        self.conv_adjust = nn.Conv2d(in_channels, feature_dim, kernel_size=1)

        # 添加额外的卷积层增强特征提取能力
        self.conv_extra = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_image):
        # 通过骨干网络提取基础特征
        x = self.features(rgb_image)

        # 调整特征维度
        x = self.conv_adjust(x)

        # 通过额外的卷积层
        x = self.conv_extra(x)

        return x


class FeaturePositionFusion(nn.Module):
    """
    特征与位置融合模块 - 整合图像特征和3D点云位置
    输入:
        image_features: 图像特征 [batch_size, feature_dim, feat_height, feat_width]
        point_cloud: 点云数据 [batch_size, num_points, 3] (x, y, z)
        camera_params: 相机参数 (字典)
    输出: 融合特征 [batch_size, num_points, feature_dim + 3]
    """

    def __init__(self, feature_dim=256):
        super(FeaturePositionFusion, self).__init__()
        self.feature_dim = feature_dim

    def project_3d_to_2d(self, point_cloud, camera_params):
        """
        将3D点云投影到2D图像平面
        输入:
            point_cloud: [batch_size, num_points, 3]
            camera_params: 包含相机内参和外参的字典
        返回:
            pixel_coords: 2D像素坐标 [batch_size, num_points, 2]
            valid_mask: 有效点掩码 [batch_size, num_points]
        """
        batch_size, num_points, _ = point_cloud.shape

        # 从相机参数中提取内参矩阵 [3, 3]
        K = camera_params['intrinsic']

        # 从相机参数中提取外参矩阵 [4, 4]
        T = camera_params['extrinsic']

        # 将点云转换为齐次坐标 [batch_size, num_points, 4]
        ones = torch.ones(batch_size, num_points, 1, device=point_cloud.device)
        points_homo = torch.cat([point_cloud, ones], dim=-1)

        # 应用外参变换 (世界坐标系 -> 相机坐标系)
        # [batch_size, 4, 4] @ [batch_size, 4, num_points] -> [batch_size, 4, num_points]
        points_cam = torch.matmul(T, points_homo.permute(0, 2, 1))

        # 转换为非齐次坐标 [batch_size, 3, num_points]
        points_cam = points_cam[:, :3, :] / points_cam[:, 3:4, :].clamp(min=1e-6)

        # 应用内参变换 (相机坐标系 -> 图像平面)
        # [batch_size, 3, 3] @ [batch_size, 3, num_points] -> [batch_size, 3, num_points]
        points_img = torch.matmul(K, points_cam)

        # 转换为像素坐标 (u, v) [batch_size, 2, num_points]
        pixel_coords = points_img[:, :2, :] / points_img[:, 2:3, :].clamp(min=1e-6)

        # 转置为 [batch_size, num_points, 2]
        pixel_coords = pixel_coords.permute(0, 2, 1)

        # 创建有效点掩码 (深度>0且在图像范围内)
        z = points_cam[:, 2, :]  # 深度值 [batch_size, num_points]
        valid_depth = (z > 0.1)  # 深度大于10cm视为有效

        # 获取图像尺寸
        img_h, img_w = camera_params['image_size']

        # 检查点是否在图像范围内
        valid_x = (pixel_coords[..., 0] >= 0) & (pixel_coords[..., 0] < img_w)
        valid_y = (pixel_coords[..., 1] >= 0) & (pixel_coords[..., 1] < img_h)

        # 组合有效掩码
        valid_mask = valid_depth & valid_x & valid_y

        return pixel_coords, valid_mask

    def forward(self, image_features, point_cloud, camera_params):
        batch_size, feature_dim, feat_height, feat_width = image_features.shape
        num_points = point_cloud.shape[1]

        # 1. 将3D点云投影到2D图像平面
        pixel_coords, valid_mask = self.project_3d_to_2d(point_cloud, camera_params)

        # 2. 将像素坐标归一化到[-1, 1]范围 (grid_sample的要求)
        # 注意: 这里归一化到原图尺寸，特征图尺寸与原图比例会自动处理
        img_h, img_w = camera_params['image_size']
        norm_pixel_coords = pixel_coords.clone()
        norm_pixel_coords[..., 0] = (norm_pixel_coords[..., 0] / (img_w - 1)) * 2 - 1  # u
        norm_pixel_coords[..., 1] = (norm_pixel_coords[..., 1] / (img_h - 1)) * 2 - 1  # v

        # 3. 提取特定位置的图像特征 (双线性插值)
        # 调整坐标形状: [batch_size, num_points, 1, 2]
        sampling_grid = norm_pixel_coords.view(batch_size, num_points, 1, 2)

        # 采样特征: [batch_size, feature_dim, num_points, 1]
        sampled_features = F.grid_sample(
            image_features,
            sampling_grid,
            align_corners=True,
            padding_mode='zeros'
        )

        # 调整形状: [batch_size, num_points, feature_dim]
        sampled_features = sampled_features.squeeze(-1).permute(0, 2, 1)

        # 4. 将无效点的特征置零
        # 扩展valid_mask维度以匹配特征维度 [batch_size, num_points, 1]
        valid_mask = valid_mask.unsqueeze(-1)
        sampled_features = sampled_features * valid_mask.float()

        # 5. 拼接图像特征和3D位置 -> [batch_size, num_points, feature_dim + 3]
        fused_points = torch.cat([sampled_features, point_cloud], dim=-1)

        return fused_points, valid_mask.squeeze(-1)


class FrontEnd(nn.Module):
    """
    前端模块 - 整合图像特征提取和位置融合
    输入:
        rgb_image: RGB图像 [batch_size, 3, H, W]
        point_cloud: 点云数据 [batch_size, num_points, 3]
        camera_params: 相机参数字典
    输出:
        fused_points: 融合特征点 [batch_size, num_points, feature_dim + 3]
        valid_mask: 有效点掩码 [batch_size, num_points]
    """

    def __init__(self, backbone_name='resnet18', pretrained=True, feature_dim=256):
        super(FrontEnd, self).__init__()
        self.img_feature_extractor = ImageFeatureExtractor(backbone_name, pretrained, feature_dim)
        self.feature_position_fusion = FeaturePositionFusion(feature_dim)

    def forward(self, rgb_image, point_cloud, camera_params):
        # 提取图像特征
        img_features = self.img_feature_extractor(rgb_image)

        # 融合图像特征与3D位置
        fused_points, valid_mask = self.feature_position_fusion(
            img_features, point_cloud, camera_params
        )

        return fused_points, valid_mask


# ================================================================
# 辅助函数和测试代码
# ================================================================

def create_dummy_camera_params(batch_size, img_size=(480, 640)):
    """
    创建虚拟相机参数用于测试
    返回: 包含内参、外参和图像尺寸的字典
    """
    params = {
        'intrinsic': torch.tensor([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=torch.float32).repeat(batch_size, 1, 1),

        'extrinsic': torch.eye(4, dtype=torch.float32).repeat(batch_size, 1, 1),

        'image_size': img_size  # (height, width)
    }
    return params


def test_frontend():
    """测试前端模块的功能"""
    # 设置参数
    batch_size = 2
    num_points = 1000
    img_size = (480, 640)  # H, W

    # 创建虚拟输入
    rgb_image = torch.randn(batch_size, 3, img_size[0], img_size[1])
    point_cloud = torch.randn(batch_size, num_points, 3)

    # 创建相机参数
    camera_params = create_dummy_camera_params(batch_size, img_size)

    # 初始化前端模块
    frontend = FrontEnd(feature_dim=128)

    # 前向传播
    fused_points, valid_mask = frontend(rgb_image, point_cloud, camera_params)

    # 打印输出形状
    print("RGB图像形状:", rgb_image.shape)
    print("点云形状:", point_cloud.shape)
    print("融合点特征形状:", fused_points.shape)  # 应为 [2, 1000, 128+3=131]
    print("有效掩码形状:", valid_mask.shape)  # 应为 [2, 1000]

    # 统计有效点比例
    valid_ratio = valid_mask.float().mean().item()
    print(f"有效点比例: {valid_ratio:.2%}")


if __name__ == "__main__":
    test_frontend()