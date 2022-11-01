sed -i 's/\s.*$//' compiler.txt 
sed -i '1!d' compiler.txt 
echo "$(cat compiler.txt)"
#cat compiler.txt
