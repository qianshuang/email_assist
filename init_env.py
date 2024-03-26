# -*- coding: utf-8 -*-

import redis
import redis_lock

r = redis.Redis()

for elem in r.keys():
    print(elem)

redis_lock.reset_all(r)

for elem in r.keys():
    print(elem)
