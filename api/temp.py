    def request_client_token(self):
        os.environ['client_id'] = input('Please Enter Clinet ID: ') # Will make this a env variable, Ideally this should be alredy set
        os.environ['client_secret'] = input('Please enter Clinet Secret: ')

        print('Opening OAuth URL to fetch credentials')

        ACCESS_URI = self._oauth_server + 'oauth/authorize?client_id=' + os.environ['client_id'] + '&scope=openid+profile&response_type=code&nonce=abc'
        run(ACCESS_URI)
        headers = {"Authorization": "Bearer " + self._client_token['access_token']}
        requests_data = requests.get(url=self._oauth_server + 'api/me', headers=headers)

        if requests_data.status_code == 200:
            requests_data = requests_data.json()
            self._user_id = requests_data['id']

    def init(self, file_path=None):
        if file_path and self._client_token is None:
            with open(file_path, 'r') as f:
                self._client_token = json.load(f)
            headers = {"Authorization": "Bearer " + self._client_token['access_token']}
            requests_data = requests.get(
                url=self._oauth_server + 'api/me', headers=headers)
            if requests_data.status_code == 401:
                print("Clinet Token Expired. Fetching new one. Please follow steps")
                self.request_client_token()
            elif requests_data.status_code == 200:
                requests_data = requests_data.json()
                self._user_id = requests_data['id']

        #TODO: Create the token request if not present and save it to the disk
