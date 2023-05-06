# CPU/Network Utilization Fetcher

- Fetches the CPU utilization from `/proc/sys` including the fractional Unix epoch in the format below
```
      seconds      cpu  user nice  system   idle   iowait  irq  softirq  steal   guest  guest_nice
1683362520.530165  cpu  2049  32    2153   117059   1089    0     13       0       0         0     
```

- If used with root, the frequency of each CPU core is stored as well (`/sys/devices/system/cpu/`)
```
      seconds      cpu  user nice  system   idle   iowait  irq  softirq  steal   guest  guest_nice  cpu0_freq  cpu1_freq  cpu2_freq  cpu3_freq
1683362520.530165  cpu  2049  32    2153   117059   1089    0     13       0       0         0         0.70      0.70        0.70       0.70
```

- Fetches the Network Utilization from `/proc/net/dev` including the fractional Unix epoch in the format below
```
                  Inter- |                            Receive                           |  Transmit
seconds            face  |  bytes    packets  errs drop fifo frame compressed multicast |  bytes     packets errs drop fifo colls carrier compressed
1683362520.530493  wlan0 :  275731     846     0    0    0     0      0         205        113278    589      0    0    0     0      0        0
```
