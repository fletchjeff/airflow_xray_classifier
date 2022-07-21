from datetime import datetime
from airflow import DAG
from airflow.operators.bash_operator import BashOperator

dag = DAG('get_ip_dag', description='Get IP DAG',
          schedule_interval=None,
          start_date=datetime(2017, 3, 20), 
          catchup=False,
        )

with dag:        
    get_ip = BashOperator(
        task_id='get_ip', 
        bash_command='curl icanhazip.com'
        )

get_ip