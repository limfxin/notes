###
 # This script evaluates plausibility of the results generated by our models
###

TASK=senti
if [[ $TASK == "mrc" ]]; then
    MODELS=("roberta_base" "roberta_large")
    MODES=("attention" "integrated_gradient")
else
    MODELS=("lstm" "roberta_base" "roberta_large")
    MODES=("attention" "integrated_gradient" "lime")
fi

for BASE_MODEL in ${MODELS[*]};
do
    for INTER_MODE in ${MODES[*]};
    do
        for LANGUAGE in "ch" "en";
        do
            GOLDEN_PATH=../golden/${TASK}_${LANGUAGE}.tsv
            PRED_PATH=../../rationale_extraction/evaluation_data/${TASK}/${BASE_MODEL}_${INTER_MODE}_${LANGUAGE}

            SAVE_PATH=res/
            [ -d $SAVE_PATH ] || mkdir -p $SAVE_PATH

            echo $BASE_MODEL$'_'$INTER_MODE$'_'$LANGUAGE

            python3 ./eval_${TASK}.py \
                --language $LANGUAGE \
                --golden_path $GOLDEN_PATH \
                --pred_path $PRED_PATH
        done
    done
done