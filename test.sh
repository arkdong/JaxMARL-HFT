export PYTHONPATH="$(pwd):$PYTHONPATH"

srun $HOME/JaxMARL-HFT/python3 gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
    --config-name="ippo_rnn_JAXMARL_2player"