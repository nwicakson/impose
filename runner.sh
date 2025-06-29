traincpn() {
    CUDA_VISIBLE_DEVICES=0 python main_diffpose_frame.py --train \
    --config human36m_diffpose_uvxyz_cpn.yml --batch_size 1024 \
    --model_pose_path checkpoints/gcn_xyz_cpn.pth \
    --doc human36m_diffpose_uvxyz_cpn --exp exp --ni \
    >exp/human36m_diffpose_uvxyz_cpn.out 2>&1 &
}

traingt() {
    python main_diffpose_frame.py --train \
    --config human36m_diffpose_uvxyz_gt.yml --batch_size 1024 \
    --model_pose_path checkpoints/gcn_xyz_gt.pth \
    --doc human36m_diffpose_uvxyz_gt --exp exp --ni \
    >exp/human36m_diffpose_uvxyz_gt.out 2>&1 &
}

trainimpose() {
    CUDA_VISIBLE_DEVICES=0 python main_idiffpose_frame.py \
    --train --config human36m_diffpose_uvxyz_deq.yml --deq_enabled --deq_middle_layer --deq_final_layer \
    --model_pose_path checkpoints/gcn_xyz_gt.pth \
    --batch_size 1024 --test_times 1 --test_timesteps 2 --test_num_diffusion_timesteps 12 \
    --doc idiffpose_deq_implicit
}

trainimpose2() {
    CUDA_VISIBLE_DEVICES=2 python main_idiffpose_frame.py \
    --train --config human36m_diffpose_uvxyz_deq_50.yml  --deq_enabled \
    --deq_iterations 20 \
    --model_pose_path checkpoints/gcn_xyz_gt.pth \
    --model_diff_path checkpoints/diffpose_uvxyz_gt.pth \
    --batch_size 1024 --test_times 1 --test_timesteps 2 --test_num_diffusion_timesteps 12 \
    --doc idiffpose_deq_20_pfo_all
}

testcpn() {
    CUDA_VISIBLE_DEVICES=0 python main_diffpose_frame.py \
    --config human36m_diffpose_uvxyz_cpn.yml --batch_size 1024 \
    --model_pose_path checkpoints/gcn_xyz_cpn.pth \
    --model_diff_path checkpoints/diffpose_uvxyz_cpn.pth \
    --doc t_human36m_diffpose_uvxyz_cpn --exp exp --ni \
    >exp/t_human36m_diffpose_uvxyz_cpn.out 2>&1 &
}

testgt() {
    CUDA_VISIBLE_DEVICES=0 python main_diffpose_frame.py \
    --config human36m_diffpose_uvxyz_gt.yml --batch_size 1024 \
    --model_pose_path checkpoints/gcn_xyz_gt.pth \
    --model_diff_path checkpoints/diffpose_uvxyz_gt.pth \
    --doc t_human36m_diffpose_uvxyz_gt --exp exp --ni \
    >exp/t_human36m_diffpose_uvxyz_gt.out 2>&1 &
}

testimpose() {
    CUDA_VISIBLE_DEVICES=2 python main_idiffpose_frame.py \
    --config human36m_diffpose_uvxyz_deq_50.yml  --deq_enabled --deq_middle_layer --deq_final_layer \
    --deq_iterations 1 \
    --model_pose_path checkpoints/gcn_xyz_gt.pth \
    --model_diff_path exp/idiffpose_deq_20_pfo_1/ckpt_best.pth \
    --batch_size 1024 --test_times 1 --test_timesteps 2 --test_num_diffusion_timesteps 12 \
    --doc t_idiffpose_deq_20_pfo_3 --exp exp --ni \
    >exp/t_idiffpose_deq_20_pfo_3.out 2>&1 &
}

# Main script
case "$1" in
    traincpn)
        traincpn
        ;;
    traingt)
        traingt
        ;;
    trainimpose)
        trainimpose
        ;;
    trainimpose2)
        trainimpose2
        ;;
    testcpn)
        testcpn
        ;;
    testgt)
        testgt
        ;;
    testimpose)
        testimpose
        ;;
    *)
        echo "Usage: $0 {traincpn|traingt|trainimpose|trainimpose2|testcpn|testgt|testimpose}"
        exit 1
esac
exit 0
