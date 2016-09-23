# LinkedIn-Python Pip package

Strengths: working well with 2.7
Weaknesses: need to make lots of changes with Python3. Notes below are what I learnt to make it work. 
Eventually, I realised that it's too restrictive to be useful for general text analysis





AttributeError: 'Service' object has no attribute 'process'


### In Linkedin.PY: Traceback (most recent call last):
  File "/Users/chengyu/PycharmProjects/MakeAltaGreatAgain/LinkedIn.py", line 1, in <module>
    from linkedin import server
  File "/Users/chengyu/anaconda/lib/python3.5/site-packages/linkedin/server.py", line 24
    print auth.authorization_url
    
Change this method in linkedin.py

add: import pickle

def _make_new_state(self):
        return hashlib.md5(
            #'%s%s' % (random.randrange(0, 2**63), self.secret)).hexdigest()
            pickle.dumps(random.randrange(0, 2**63))+
            pickle.dumps(self.secret)
            ).hexdigest()

### params = urlparse.parse_qs(p[1], True, True) AttributeError: 'function' object has no attribute 'parse_qs'
Change the line to: params =urllib.parse.parse_qs(p[1], True, True)

### LinkedIn is not authorised to ...:
 Edit the permissions constant in LinkedIn.py
 
 
### TypeError: string argument expected, got 'bytes'
linkedin.py: 

  with contextlib.closing(StringIO()) as result:
              if type(selector) == dict:
                  for k, v in list(selector.items()):
 -                    result.write('%s:(%s)' % (to_utf8(k), cls.parse(v)))
 +                    result.write('%s:(%s)' % (k, cls.parse(v)))
              elif type(selector) in (list, tuple):
                  result.write(','.join(map(cls.parse, selector)))
              else:
 -                result.write(to_utf8(selector))
 +                result.write(selector)
              return result.getvalue()
Details can be found here: https://github.com/andrewychoi/python3-linkedin/commit/76b577448460f21949084da26ba19616772c2222

