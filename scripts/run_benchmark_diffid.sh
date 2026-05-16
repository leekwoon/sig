set -e

bash scripts/benchmark_diffid/agi.sh
bash scripts/benchmark_diffid/big.sh
bash scripts/benchmark_diffid/eig.sh
bash scripts/benchmark_diffid/gig.sh
bash scripts/benchmark_diffid/grad_input.sh
bash scripts/benchmark_diffid/ig.sh
bash scripts/benchmark_diffid/ig2.sh
bash scripts/benchmark_diffid/mig.sh
bash scripts/benchmark_diffid/samp.sh

bash scripts/benchmark_diffid/spectral_ig.sh 0.1
bash scripts/benchmark_diffid/spectral_ig.sh 0.2
bash scripts/benchmark_diffid/spectral_ig.sh 0.3
bash scripts/benchmark_diffid/spectral_ig.sh 0.4
bash scripts/benchmark_diffid/spectral_ig.sh 0.5
bash scripts/benchmark_diffid/spectral_ig.sh 0.6
