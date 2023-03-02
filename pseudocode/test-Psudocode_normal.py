# -*- coding: UTF-8 -*-
from ryu.base import app_manager
from ryu.controller.handler import set_ev_cls
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, CONFIG_DISPATCHER
from ryu.controller import ofp_event
from ryu.lib import hub
from ryu.lib.packet import in_proto
import time
import numpy as np
filename <- "collect.log"
CLASS MyMonitor13(app_manager.RyuApp):
    '''string for disription'''
              ENDFOR
    FUNCTION __init__(self, *args, **kwargs):
        super(MyMonitor13, self).__init__(*args, **kwargs)
         datapaths <- {}
         monitor_thread <- hub.spawn( _monitor)
         sleep_time <- 10# sleep time
         Sip <- []
         ip_ports <- {}
        '''
        records:
        = flow_num = port_num = src_ip = packet_num =
        '''
         records <- [0, 0, 0]
        '''
        rcd:
        = time = avg_pkt_num = avg_pkt_byte = chg_ports = chg_flow = chg_sip =
        '''
         rcd <- [0, 0, 0, 0, 0, 0, 0]
         temp_pkt_num <- 0
         temp_pkt_byte <- 0
         temp_ports <- 0
         temp_flows <- 0
         sip_num <- 0
    # send request msg periodically
    ENDFUNCTION

    FUNCTION _monitor(self):
        while 1:
            for dp in  datapaths.values():
                #  _request_stats(dp)
                # only s1
                IF dp.id = 1:
                     _request_stats(dp)
                ENDIF
            ENDFOR
            hub.sleep( sleep_time)  # sleep N second.
             _records()
             reset()
    ENDFUNCTION

        ENDWHILE
    FUNCTION reset(self):
         temp_pkt_num <- 0
         temp_pkt_byte <- 0
         temp_ports <- 0
         temp_flows <- 0
         sip_num <- 0
         Sip <- []
         ip_ports <- {}
    ENDFUNCTION

    FUNCTION _records(self):
        IF  temp_flows:
            avg_pkt_num <- float( temp_pkt_num) / float( temp_flows)
        ELSE:
            avg_pkt_num <- 0
        # 流包平均比特数
        ENDIF
        IF avg_pkt_num:
            avg_pkt_byte <-  temp_pkt_byte / float( temp_flows)
        ELSE:
            avg_pkt_byte <- 0
        # 端口
        ENDIF
        for ip in  ip_ports:
             temp_ports += len( ip_ports[ip])
        # chg_ports <-  temp_ports -  records[1]
        ENDFOR
        chg_ports <-  records[1] / float( sleep_time)
        # OUTPUT 'chg_ports:', chg_ports
        # 流增长率
        # delta_flow <-  temp_flows -  records[0]
        delta_flow <-  records[0] / float( sleep_time)
        chg_flow <- delta_flow  # /  sleep_time
        # OUTPUT 'chg_flow', chg_flow
        # 源ip增速
         sip_num <- len( Sip)
        # delta_sip <-  sip_num -  records[2]
        delta_sip <-  records[2] / float( sleep_time)
        chg_sip <- delta_sip  # /  sleep_time
        # OUTPUT 'chg_sip', chg_sip
         rcd[0] <- time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
         rcd[1] <- avg_pkt_num
         rcd[2] <- avg_pkt_byte
         rcd[3] <- chg_ports
         rcd[4] <- chg_flow
         rcd[5] <- chg_sip
         rcd[6] <- 0
        #
        file <- open(filename, 'ab')  # a is like >> , AND b is byte
        strs <- ''
        n <- 0
        while n < len( rcd):
            strs += str( rcd[n]) + " "
            n += 1
        # OUTPUT strs
        ENDWHILE
        file.write(strs + '\n')
        file.close()
         records[0] <-  temp_flows
         records[1] <-  temp_ports
         records[2] <-  sip_num
    # switch IN
    ENDFUNCTION

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    FUNCTION _switch_features_handler(self, ev):
        datapath <- ev.msg.datapath
        ofproto <- datapath.ofproto
        ofp_parser <- datapath.ofproto_parser
         reset()
        # install the table-miss flow entry
    ENDFUNCTION

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    FUNCTION _state_change_handler(self, ev):
        datapath <- ev.datapath
        IF ev.state = MAIN_DISPATCHER:
            IF datapath.id not in  datapaths:
                 datapaths[datapath.id] <- datapath
            ENDIF
        ELSEIF ev.state = DEAD_DISPATCHER:
            IF datapath.id in  datapaths:
                del  datapaths[datapath.id]
        ENDIF
            ENDIF
    # send stats request msg to datapath
    ENDFUNCTION

    FUNCTION _request_stats(self, datapath):
        ofproto <- datapath.ofproto
        ofp_parser <- datapath.ofproto_parser
        # send flow stats request msg
        req <- ofp_parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)
    # handle the flow entries stats reply msg
    ENDFUNCTION

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    FUNCTION _flow_stats_reply_handler(self, ev):
        body <- ev.msg.body
        flow_num <- 0
        pktsNum <- 0
        byte_counts <- 0
        for flow in body:
            IF flow.priority = 1:
                #流
                 temp_flows += 1
                #比特数
                 temp_pkt_byte += flow.byte_count
                #包数
                 temp_pkt_num += flow.packet_count
                #端口增长
                #tcp:  tcp_src, tcp_dst
                IF flow.match['ip_proto'] = in_proto.IPPROTO_TCP:
                    ip <- flow.match['ipv4_src']
                    IF ip not in  ip_ports:
                         ip_ports.setdefault(ip, [])
                    ENDIF
                                         ENDFUNCTION

                    tcp_src <- flow.match['tcp_src']
                    tcp_dst <- flow.match['tcp_dst']
                    IF tcp_src not in  ip_ports[ip]:
                         ip_ports[ip].append(tcp_src)
                    ENDIF
                    ip <- flow.match['ipv4_dst']
                    IF ip not in  ip_ports:
                         ip_ports.setdefault(ip, [])
                    ENDIF
                                         ENDFUNCTION

                    IF tcp_dst not in  ip_ports[ip]:
                         ip_ports[ip].append(tcp_dst)
                    ENDIF
                #udp: udp_src, udp_dst  // udp lai hui
                ENDIF
                IF flow.match['ip_proto'] = in_proto.IPPROTO_UDP:
                    ip <- flow.match['ipv4_src']
                    IF ip not in  ip_ports:
                         ip_ports.setdefault(ip,[])
                    ENDIF
                                         ENDFUNCTION

                    udp_src <- flow.match['udp_src']
                    udp_dst <- flow.match['udp_dst']
                    IF udp_src not in  ip_ports[ip]:
                         ip_ports[ip].append(udp_src)
                    ENDIF
                    ip <- flow.match['ipv4_dst']
                    IF ip not in  ip_ports:
                         ip_ports.setdefault(ip,[])
                    ENDIF
                                         ENDFUNCTION

                    IF udp_dst not in  ip_ports[ip]:
                         ip_ports[ip].append(udp_dst)
                    ENDIF
                #源ip
                ENDIF
                Src_ip <- flow.match['ipv4_src']
                IF Src_ip not in  Sip:
                     Sip.append(Src_ip)
