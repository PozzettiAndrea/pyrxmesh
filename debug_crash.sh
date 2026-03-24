#!/bin/bash
# Run compute-sanitizer in a loop until we catch the crash
for i in $(seq 1 10); do
    echo "=== Run $i ==="
    compute-sanitizer --tool memcheck --log-file /tmp/sanitizer_run_${i}.log python3.13 -c "
import pyrxmesh; pyrxmesh.init()
v, f = pyrxmesh.load_obj('RXMesh/input/dragon.obj')
el = pyrxmesh.expected_edge_length(v, f)
pyrxmesh.feature_remesh(v, f,
    relative_len=el.target_edge_length/el.avg_edge_length,
    iterations=15, max_passes=1, verbose=True)
print('OK')
" 2>&1 | tail -5

    # Check if sanitizer found errors
    if grep -q "Invalid" /tmp/sanitizer_run_${i}.log 2>/dev/null; then
        echo "=== FOUND ERROR in run $i ==="
        grep "Invalid" /tmp/sanitizer_run_${i}.log | head -10
        echo "Full log: /tmp/sanitizer_run_${i}.log"
        break
    fi

    if grep -q "CUDA ERROR" /tmp/sanitizer_run_${i}.log 2>/dev/null; then
        echo "=== FOUND CUDA ERROR in run $i ==="
        cat /tmp/sanitizer_run_${i}.log | head -30
        break
    fi

    echo "Run $i clean, retrying..."
done
