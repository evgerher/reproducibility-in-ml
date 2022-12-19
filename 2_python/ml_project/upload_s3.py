import boto3

session = boto3.session.Session(profile_name='ydata-demo')

s3 = session.client(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net',
)

BUCKET_NAME = 'evgerher-ydata-demo'

if __name__ == '__main__':
    s3.upload_file('artifacts/model_lr.pkl', BUCKET_NAME, 'artifacts/lr.pkl')
