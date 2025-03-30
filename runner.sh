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
    CUDA_VISIBLE_DEVICES=2 python main_diffpose_frame.py \
    --train --implicit_layers \
    --config human36m_diffpose_uvxyz_gt.yml --batch_size 1024 \
    --debug \
    --doc ipose --exp exp --ni \
    >exp/ipose.out 2>&1 & \
    
}

trainipose2() {
    CUDA_VISIBLE_DEVICES=2 python main_implicit_pose.py \
    --config human36m_ipose.yml \
    --doc ipose_dynamicmemory \
    --use_implicit \
    --use_dynamic_chunks \
    --min_chunk_size 256 \
    --max_chunk_size 1024 \
    --target_memory_usage 0.9 \
    --implicit_iters 20 \
    --implicit_tol 1e-5 \
    --min_iterations 10 \
    --track_metrics
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

# Main script
case "$1" in
    traincpn)
        traincpn
        ;;
    traingt)
        traingt
        ;;
    trainipose)
        trainipose
        ;;
    trainipose2)
        trainipose2
        ;;
    testcpn)
        testcpn
        ;;
    testgt)
        testgt
        ;;
    *)
        echo "Usage: $0 {traincpn|traingt|trainipose|trainipose2|testcpn|testgt}"
        exit 1
esac
exit 0
