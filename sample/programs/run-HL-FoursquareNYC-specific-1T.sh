#!/bin/bash
BASE="/Users/tarlisportela/workdir/programs/mat-analysis-pkg/matanalysis"
DATAPATH="${BASE}/sample/data/multiple_trajectories/FoursquareNYC"
RESPATH="${BASE}/results/EXP01-1T_7G"
MNAME="HL-specific"

for RUN in "run1" "run2" "run3" "run4" "run5"
do
DIR="${RESPATH}/FoursquareNYC/${RUN}"
if [ -d "${DIR}/${MNAME}" ]; then
   echo "${DIR}/${MNAME} ... [OK]"
else

mkdir -p "${DIR}/${MNAME}"

java -Xmx7G -jar "./sample/programs/HIPERMovelets.jar" -curpath "${DATAPATH}/${RUN}" -respath "${DIR}/${MNAME}" -descfile "${BASE}/sample/data/multiple_trajectories/descriptors/FoursquareNYC_specific_hp.json" -nt 1 -version hiper -ms -1 -Ms -3 -TC 7d  2>&1 | tee -a "${DIR}/${MNAME}/${MNAME}.txt" 

MAT-MergeDatasets.py "${DIR}/${MNAME}"

# --------------------------------------------------------------------------------------
MAT-MC.py -c "MLP,RF" "${DIR}" "${MNAME}"

echo "${DIR}/${MNAME} ... Done."
fi
done
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
# Automatize - END generated script
