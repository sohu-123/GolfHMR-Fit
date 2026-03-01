import torch

from sam_3d_body.models.heads.mhr_head import (
    bilateral_hand_rotation_loss,
    wrist_forearm_axis_loss,
    wrist_angle_soft_limit,
    rotation_matrix_to_angle,
)


def test_rotation_matrix_to_angle_and_wrist_losses():
    B = 4
    N = 130
    # Create identity rotations
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)

    # Angles should be zero for identity
    angles = rotation_matrix_to_angle(R[:, 0])
    assert angles.shape == (B,)
    assert torch.allclose(angles, torch.zeros_like(angles), atol=1e-6)

    # Test bilateral loss (identity -> zero)
    loss_bilateral = bilateral_hand_rotation_loss(R, 10, 20)
    assert torch.isclose(loss_bilateral, torch.tensor(0.0))

    # Create simple joints_3d: elbow at origin, wrist at x=1, hand at x=2
    joints = torch.zeros(B, N, 3)
    joints[:, 5, 0] = 0.0  # elbow idx
    joints[:, 6, 0] = 1.0  # wrist idx
    joints[:, 7, 0] = 2.0  # hand idx

    loss_axis = wrist_forearm_axis_loss(joints, 5, 6, 7)
    # forearm and hand align so loss close to 0
    assert loss_axis < 1e-6

    # Test wrist angle soft limit with small rotation
    angle_small = 0.5
    # Construct a rotation matrix with small rotation about z axis
    cos = torch.cos(torch.tensor(angle_small))
    sin = torch.sin(torch.tensor(angle_small))
    Rz = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    R2 = R.clone()
    R2[:, 12] = Rz

    loss_wrist = wrist_angle_soft_limit(R2, 12, max_angle_rad=1.2)
    assert loss_wrist >= 0.0


if __name__ == '__main__':
    test_rotation_matrix_to_angle_and_wrist_losses()
    print('hand constraints tests passed')
