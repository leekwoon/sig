set -e

bash scripts/benchmark_roar/agi.sh
bash scripts/benchmark_roar/big.sh
bash scripts/benchmark_roar/eig.sh
bash scripts/benchmark_roar/gig.sh
bash scripts/benchmark_roar/grad_input.sh
bash scripts/benchmark_roar/ig.sh
bash scripts/benchmark_roar/ig2.sh
bash scripts/benchmark_roar/mig.sh
bash scripts/benchmark_roar/random.sh
bash scripts/benchmark_roar/samp.sh

bash scripts/benchmark_roar/spectral_ig.sh
