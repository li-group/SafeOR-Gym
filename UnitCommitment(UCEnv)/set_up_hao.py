# local
export remoteIP="ec2-user@ec2-3-236-110-40.compute-1.amazonaws.com"
ssh -i "./hao_pocs.pem" $remoteIP

export remoteIP="ec2-user@ec2-44-200-193-224.compute-1.amazonaws.com"
ssh -i "./hao_ldr.pem" $remoteIP

export remoteIP="ec2-user@ec2-44-221-42-233.compute-1.amazonaws.com"
ssh -i "./Hao_test.pem" $remoteIP




# --- bash on the remote instance ---
cd /home/ec2-user/;
# Install pipaws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A
curl -O https://bootstrap.pypa.io/get-pip.py;
python3 get-pip.py --user;
# Install dependencies...
sudo yum install git -y

git clone https://github.com/GeraltSHEN/omnisafe.git
cd omnisafe
pip install setuptools==65.5.1
pip install .

# local
scp -i "./Hao_test.pem" -r ./GitHub/SafeRL_Environments/UnitCommitment "${remoteIP}:/home/ec2-user/"

# Perform other operation;s...
# screen / screen -r / ctrl+A ctrl+D
exit;


scp -i "./Hao_test.pem" -r "${remoteIP}:/home/ec2-user/SafeRL_Environments/UnitCommitment/exp-x" ~/Downloads

scp -i "./Hao_test.pem" "${remoteIP}:/home/ec2-user/SafeRL_Environments/UnitCommitment/" ~/Downloads
# --- powershell ---


export remoteIP="ec2-user@ec2-3-236-110-40.compute-1.amazonaws.com"
scp -i "./hao_pocs.pem" -r "${remoteIP}:/home/ec2-user/Learn2Proj/models" ~/Downloads/
scp -i "./hao_pocs.pem" -r "${remoteIP}:/home/ec2-user/Learn2Proj/logs" ~/Downloads/
scp -i "./hao_pocs.pem" -r "${remoteIP}:/home/ec2-user/Learn2Proj/data/results_summary" ~/Downloads/

export remoteIP="ec2-user@ec2-44-200-193-224.compute-1.amazonaws.com"
scp -i "./hao_ldr.pem" -r "${remoteIP}:/home/ec2-user/Learn2Proj/models" ~/Downloads/
scp -i "./hao_ldr.pem" -r "${remoteIP}:/home/ec2-user/Learn2Proj/logs" ~/Downloads/
scp -i "./hao_ldr.pem" -r "${remoteIP}:/home/ec2-user/Learn2Proj/data/results_summary" ~/Downloads/

export remoteIP="ec2-user@ec2-44-221-42-233.compute-1.amazonaws.com"
scp -i "./Hao_test.pem" -r "${remoteIP}:/home/ec2-user/Learn2Proj/models" ~/Downloads/
scp -i "./Hao_test.pem" -r "${remoteIP}:/home/ec2-user/Learn2Proj/logs" ~/Downloads/
scp -i "./Hao_test.pem" -r "${remoteIP}:/home/ec2-user/Learn2Proj/data/results_summary" ~/Downloads/


