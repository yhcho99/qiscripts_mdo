# #!/bin/bash

# array=( test_nvq_value70_intangible30_1 test_nvq_value70_intangible30_2 test_nvq_value70_intangible30_3  )

# for i in "${array[@]}"

# do
#     python3 experiment/nvq-argparser.py --identifier $i --value 0.7 --intangible 0.3
# done


# array=( test_nvq_value60_intangible40_1 test_nvq_value60_intangible40_2 test_nvq_value60_intangible40_3  )

# for i in "${array[@]}"

# do
#     python3 experiment/nvq-argparser.py --identifier $i --value 0.6 --intangible 0.4
# done


# array=( test_nvq_value50_intangible50_1 test_nvq_value50_intangible50_2 test_nvq_value50_intangible50_3 )

# for i in "${array[@]}"

# do
#     python3 experiment/nvq-argparser.py --identifier $i --value 0.5 --intangible 0.5
# done

# array=( test_nvq_value40_intangible60_1 test_nvq_value40_intangible60_2 test_nvq_value40_intangible60_3 )

# for i in "${array[@]}"

# do
#     python3 experiment/nvq-argparser.py --identifier $i --value 0.4 --intangible 0.6
# done


# array=( test_nvq_value30_intangible70_1 test_nvq_value30_intangible70_2 test_nvq_value30_intangible70_3 )

# for i in "${array[@]}"

# do
#     python3 experiment/nvq-argparser.py --identifier $i --value 0.3 --intangible 0.7
# done


#!/bin/bash

array=( test_nvq_value30_intangible70_1 )

for i in "${array[@]}"

do
    python3 experiment/nvq-argparser.py --identifier $i --value 0.3 --intangible 0.7
done

