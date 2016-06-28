
#!/bin/bash
#bash run_test.sh >& out
grep "coordinates mismatch" out >& coords_mismatch
sed -e '1,1d' < coords_mismatch >&  cm
mv cm coords_mismatch
sed 's/^.*:/:/' coords_mismatch >& mismatches
grep "thickness mismatch" out >& thk_mismatch
sed -e '1,1d' < thk_mismatch >& tm
mv tm thk_mismatch
sed 's/^.*:/:/' thk_mismatch >> mismatches
grep "surface heigth mismatch" out >& sh_mismatch
sed -e '1,1d' < sh_mismatch >& sh
mv sh sh_mismatch
sed 's/^.*:/:/' sh_mismatch >> mismatches
grep "basal_friction mismatch" out >& beta_mismatch
sed -e '1,1d' < beta_mismatch >& beta
mv beta beta_mismatch
sed 's/^.*:/:/' beta_mismatch >> mismatches
grep "x comp of velocity mismatch" out >& uvel_mismatch
sed -e '1,1d' < uvel_mismatch >& uvel
mv uvel uvel_mismatch
sed 's/^.*:/:/' uvel_mismatch >> mismatches
grep "y comp of velocity mismatch" out >& vvel_mismatch
sed -e '1,1d' < vvel_mismatch >& vvel
mv vvel vvel_mismatch
sed 's/^.*:/:/' vvel_mismatch >> mismatches
grep "temperature mismatch" out >& temp_mismatch
sed -e '1,1d' < temp_mismatch >& temp
mv temp temp_mismatch
sed 's/^.*:/:/' temp_mismatch >> mismatches
sed -i -e 's/://g' mismatches
sed -i -e 's/ //g' mismatches
