const assert = require('node:assert/strict');
const {patternGeometry} = require('../../caisen/static/pattern-chart.js');
const bars=Array.from({length:300},(_,i)=>({date:new Date(Date.UTC(2025,0,1+i)).toISOString().slice(0,10),close:10}));
const shape={pattern_type:'w_bottom',state:'target_reached',key_points:{pivot_1_index:4,pivot_1:8,pivot_2_index:20,pivot_2:12,pivot_3_index:34,pivot_3:9},neckline:{slope:.01,intercept:12},confirmation_date:bars[60].date,targets:[16],evidence:[{code:'target_reached',value:{index:70,price:16,date:bars[90].date}}]};
const before=JSON.stringify(shape),g=patternGeometry({ _bars:bars,_chart_offset:20},shape);
assert.deepEqual(g.anchors.map(p=>p.index),[24,40,54]);
assert.deepEqual(g.anchors.map(p=>p.name),['左底','颈线拐点','右底']);
assert.equal(g.end,90);assert(g.first<24&&g.last>90);assert(g.last<180);
assert.equal(g.markers.find(p=>p.name==='目标已达').price,16);
assert.equal(g.markers.find(p=>p.name==='价格确认').index,60);
assert.equal(g.neckline[0].price,12.04);assert.equal(JSON.stringify(shape),before);
for(const [code,state] of [['structure_invalidation','invalidated'],['followthrough_confirmed','followthrough_confirmed']]){
 const p={...shape,state,evidence:[{code,value:{index:70,date:bars[90].date,level:7}}]};
 assert.equal(patternGeometry({_bars:bars,_chart_offset:20},p).markers.at(-1).price,7);
}
assert.throws(()=>patternGeometry({_bars:bars,_chart_offset:400},shape),/缺少/);
console.log('historical pattern geometry: anchors, dates, terminal states, viewport, immutability passed');
const tri={...shape,pattern_type:'triangle_bottom'};
assert.equal(patternGeometry({_bars:bars,_chart_offset:20},tri).boundaries.length,2);
const box={...shape,pattern_type:'spring',evidence:[...shape.evidence,{code:'box_upper',value:14},{code:'box_lower',value:8}]};
assert.equal(patternGeometry({_bars:bars,_chart_offset:20},box).boundaries.length,2);
const broken={...shape,state:'invalidated',evidence:[{code:'data_discontinuity',value:{index:70,date:bars[90].date,level:9}}]};
assert.equal(patternGeometry({_bars:bars,_chart_offset:20},broken).markers.at(-1).name,'行情分段结束');

const main={pattern_type:'w_bottom',state:'confirmed',chart_offset:100,
 key_points:{pivot_1_index:2,pivot_1:8,pivot_2_index:20,pivot_2:12,pivot_3_index:38,pivot_3:9},
 evidence:[
  {code:'main_structure',value:{role:'main',family:'w_bottom',support_lower:8,support_upper:9,neckline:12}},
  {code:'neckline_tests',value:[{group_id:'touch-1',pivot_date:bars[125].date,known_date:bars[130].date,level:12,valid:true},
   {group_id:'touch-2',pivot_date:bars[132].date,known_date:bars[137].date,level:12.1,valid:true}]},
  {code:'neckline_breakout',value:{date:bars[150].date,close:12.5,frozen_neckline:12,required_close:12.2}}
 ]};
const mg=patternGeometry({_bars:bars,_chart_offset:20},main);
assert.deepEqual(mg.anchors.map(p=>p.index),[102,120,138]);
assert.deepEqual(mg.markers.filter(p=>p.name.includes('颈线测试')).map(p=>p.index),[125,132]);
assert.equal(mg.markers.filter(p=>p.name.includes('颈线测试')).length,2);
assert.equal(mg.markers.find(p=>p.name.includes('确认')).index,150);
assert.equal(mg.markers.find(p=>p.name.includes('确认')).price,12.5);
console.log('main-bottom chart maps actual dates, grouped neckline tests, and frozen breakout');
const mainW={...main,chart_offset:100,key_points:{pivot_1_index:2,pivot_1_date:bars[102].date,pivot_1:8,
 pivot_2_index:38,pivot_2_date:bars[138].date,pivot_2:9},evidence:[{code:'main_structure',value:{role:'main',support_lower:8,support_upper:9}}]};
assert.deepEqual(patternGeometry({_bars:bars,_chart_offset:20},mainW).anchors.map(p=>p.name),['左底','右底']);
