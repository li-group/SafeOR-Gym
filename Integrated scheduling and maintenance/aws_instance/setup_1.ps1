# --- powershell ---

Set-Variable remoteIP "ec2-user@ec2-44-221-50-80.compute-1.amazonaws.com"
ssh -i "./akshdeep_safe.pem" $remoteIP

# --- bash on the remote instance ---

# cd /home/ec2-user/;
# # Install pipaws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A
# curl -O https://bootstrap.pypa.io/get-pip.py;
# python3 get-pip.py --user;
# # Install dependencies...
# sudo yum install git -y

# git clone https://github.com/li-group/omnisafe.git
# cd omnisafe
# pip install setuptools==65.5.1
# pip install .

# scp -i "./Asha_5.pem" C:\Users\aramanuj\main_fin1.py "${remoteIP}:/home/ec2-user/"
# scp -i "./Asha_5.pem" C:\Users\aramanuj\action_sample_base_1.json "${remoteIP}:/home/ec2-user/"
# # Perform other operation;s...
# # screen / screen -r / ctrl+A ctrl+D
# exit;


# scp -i "./Asha_5.pem" -r "${remoteIP}:/home/ec2-user/exp-x" C:\Users\aramanuj\omnisafe_results_1
# scp -i "./Asha_5.pem" -r "${remoteIP}:/home/ec2-user/algo---Capacity-Expansion---8f1f87fd458dda8e491cc24b0e6f0317670b701326f94010341b5638daf050a6.png" C:\Users\aramanuj\omnisafe_results_1
# # --- powershell ---

