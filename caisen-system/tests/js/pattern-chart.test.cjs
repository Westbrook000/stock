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
const segmented={...main,chart_offset:100,confirmation_date:bars[150].date,evidence:[...main.evidence,{code:'neckline_selection',value:{
 selected_candidate_id:'selected',selected_line:{slope:0,intercept:12},effective_start_date:bars[125].date,
 freeze_date:bars[137].date,line_frozen_at:bars[131].date,coverage:{end_date:bars[131].date},
 candidates:[{rank:1,selected:true,line:{slope:0,intercept:12},coverage_start_date:bars[125].date,coverage_end_date:bars[131].date},
  {rank:2,selected:false,line:{slope:0,intercept:13},coverage_start_date:bars[140].date,coverage_end_date:bars[145].date}],retired_segments:[]}}]};
const sg=patternGeometry({_bars:bars,_chart_offset:20},segmented);
assert(sg.boundaries.some(item=>item.name==='已验证压力区间'&&item.points[0].index===125&&item.points[1].index===131));
assert(sg.boundaries.some(item=>item.name==='冻结颈线延伸'&&item.points[0].index===131&&item.points[1].index===150));
assert(sg.boundaries.some(item=>item.name==='备选压力线 2'));
console.log('main-bottom chart separates validated pressure range, frozen extension, and alternatives');
const mainW={...main,chart_offset:100,key_points:{pivot_1_index:2,pivot_1_date:bars[102].date,pivot_1:8,
 pivot_2_index:38,pivot_2_date:bars[138].date,pivot_2:9},evidence:[{code:'main_structure',value:{role:'main',support_lower:8,support_upper:9}}]};
assert.deepEqual(patternGeometry({_bars:bars,_chart_offset:20},mainW).anchors.map(p=>p.name),['左底','右底']);

const joint={...segmented,state:'invalidated',evidence:[...segmented.evidence,
 {code:'joint_lifecycle',value:{instance_id:'instance-a',line_known_at:bars[131].date,
  boundary_version_id:'boundary-a',instance_discovered_at:bars[145].date,
  paired_at:bars[145].date,paired_at_status:'observed',
  boundary:{slope:0,intercept:11},state_at:'invalidated',events:[
   {instance_id:'instance-a',boundary_version_id:'boundary-a',kind:'paired',occurred_at:bars[145].date,recorded_at:bars[145].date},
   {instance_id:'instance-a',boundary_version_id:'boundary-a',kind:'confirmed',occurred_at:bars[150].date,recorded_at:bars[150].date},
   {instance_id:'instance-a',boundary_version_id:'boundary-a',kind:'invalidated',occurred_at:bars[160].date,recorded_at:bars[160].date},
   {instance_id:'other-instance',kind:'confirmed',occurred_at:bars[170].date,recorded_at:bars[170].date}
  ]}}]};
const jg=patternGeometry({_bars:bars,_chart_offset:20},joint);
assert.equal(jg.boundaries.find(item=>item.name==='已验证压力区间').points[0].price,11,
 'the selected lifecycle instance boundary overrides legacy geometry');
assert.equal(jg.boundaries.find(item=>item.name==='颈线延伸（边界已知 2025-05-12）').points[0].index,131,
 'line knowledge date, not the later paired date, anchors the extension');
assert(!jg.markers.some(item=>item.index===170), 'events from other instances must not appear');
assert(jg.markers.some(item=>item.name.includes('边界已知 2025-05-12')));
assert(jg.markers.some(item=>item.name.includes('实例配对 2025-05-26')));
assert(jg.markers.some(item=>item.name.includes('形态确认 · 发生 2025-05-31 · 记录 2025-05-31')));
assert(jg.markers.some(item=>item.name.includes('结构失效 · 发生 2025-06-10 · 记录 2025-06-10')));
assert(!jg.markers.some(item=>item.name.includes('备选压力线')),
 'alternatives do not produce selected-instance confirmation markers');
const jointTriangle=patternGeometry({_bars:bars,_chart_offset:20},{...joint,pattern_type:'triangle_bottom'});
assert(jointTriangle.boundaries.some(item=>item.name==='另一侧结构边界'),
 'the triangle structural side remains visible with lifecycle data');
const lifeWithoutBoundary={...segmented,evidence:[...segmented.evidence,
 {code:'joint_lifecycle',value:{instance_id:'instance-no-boundary',line_known_at:bars[95].date,
  invalidated_at:bars[160].date,events:[]}}]};
const noBoundaryGeometry=patternGeometry({_bars:bars,_chart_offset:20},lifeWithoutBoundary);
assert(!noBoundaryGeometry.boundaries.some(item=>item.name==='已验证压力区间'),
 'a present lifecycle without its boundary must not silently fall back to the legacy line');
assert(!noBoundaryGeometry.boundaries.some(item=>item.name==='冻结颈线'),
 'main-structure fallback lines are disabled when lifecycle exists');
assert.equal(noBoundaryGeometry.start,95,'viewport start includes lifecycle markers before the first anchor');
assert(noBoundaryGeometry.first<95);
assert(noBoundaryGeometry.markers.some(item=>item.name.includes('结构失效 · 发生 2025-06-10')),
 'top-level lifecycle terminal dates are chartable');
const legacyJoint={...segmented,evidence:segmented.evidence.filter(item=>item.code!=='joint_lifecycle')};
assert(patternGeometry({_bars:bars,_chart_offset:20},legacyJoint).boundaries.some(item=>item.name==='备选压力线 2'),
 'legacy records retain their existing neckline-selection fallback');

const expiredConflict={...segmented,state:'expired',confirmation_date:bars[250].date,evidence:[...segmented.evidence,
 {code:'neckline_breakout',value:{date:bars[250].date,close:14,frozen_neckline:12,required_close:12.2}},
 {code:'weak_crossings',value:[{date:bars[251].date,close:14.1}]},
 {code:'joint_lifecycle',value:{instance_id:'instance-expired',boundary_version_id:'boundary-a',
  line_known_at:bars[131].date,paired_at:bars[145].date,instance_discovered_at:bars[145].date,
  boundary:{slope:0,intercept:12},events:[
   {instance_id:'instance-expired',boundary_version_id:'boundary-a',kind:'weak_crossing',
    occurred_at:bars[180].date,recorded_at:bars[181].date,close:12.1},
   {instance_id:'instance-expired',boundary_version_id:'boundary-a',kind:'rejected_too_early',
    occurred_at:bars[190].date,recorded_at:bars[195].date,reason:'too_early'},
   {instance_id:'instance-expired',boundary_version_id:'boundary-a',kind:'expired',
    occurred_at:bars[200].date,recorded_at:bars[201].date},
   {instance_id:'instance-expired',boundary_version_id:'other-boundary',kind:'confirmed',
    occurred_at:bars[205].date,recorded_at:bars[205].date},
   {instance_id:'different-instance',boundary_version_id:'boundary-a',kind:'confirmed',
    occurred_at:bars[210].date,recorded_at:bars[210].date}
  ]}}]};
const expiredGeometry=patternGeometry({_bars:bars,_chart_offset:20},expiredConflict);
assert(!expiredGeometry.markers.some(item=>item.name==='价格确认'||item.name==='颈线收盘确认'),
 'stale pattern confirmation and breakout evidence cannot override an expired lifecycle');
assert(!expiredGeometry.markers.some(item=>item.name.includes('弱穿越')&&item.index===251),
 'legacy weak-crossing evidence is ignored when lifecycle is present');
assert(expiredGeometry.markers.some(item=>item.name.includes('弱穿越')&&item.index===180));
assert(expiredGeometry.markers.some(item=>item.name.includes('过早突破拒绝 · 发生 2025-07-10 · 记录 2025-07-15')),
 'historical rejection preserves occurrence and recording dates without presenting as confirmation');
assert(expiredGeometry.markers.some(item=>item.name.includes('形态过期 · 发生 2025-07-20 · 记录 2025-07-21')));
assert(!expiredGeometry.markers.some(item=>item.index===205||item.index===210),
 'confirmation from another boundary or instance is ignored');
assert.equal(expiredGeometry.end,200,'viewport bounds include lifecycle markers, not stale legacy confirmation');
assert(expiredGeometry.last>expiredGeometry.end&&expiredGeometry.last<250);
assert(!expiredGeometry.boundaries.some(item=>item.name.startsWith('备选压力线')||item.name.startsWith('失效压力段')),
 'joint lifecycle charts keep alternative and retired lines out of the main view');

// 600715's historical triangle stores full-series indices. The API's
// recent-five-year offset must not shift its frozen upper edge.
const fullHistory=Array.from({length:7037},(_,i)=>({
 date:new Date(Date.UTC(2000,0,1)+i*86400000).toISOString().slice(0,10),close:2.5
}));
const triangleLine={slope:-0.0002836879432624116,intercept:4.252482269503547};
const kp={};
[[3664,1.86],[3710,3.20],[3805,1.92],[3992,3.12],[4095,2.17]].forEach(([index,price],i)=>{
 kp[`pivot_${i+1}_index`]=index;
 kp[`pivot_${i+1}_date`]=fullHistory[index].date;
 kp[`pivot_${i+1}`]=price;
});
const historical={pattern_type:'triangle_bottom',chart_offset:5828,key_points:kp,evidence:[
 {code:'main_structure',value:{role:'main'}},
 {code:'neckline_selection',value:{effective_start_date:fullHistory[3710].date,
  coverage:{end_date:fullHistory[3996].date}}},
 {code:'joint_lifecycle',value:{boundary:triangleLine,line_known_at:fullHistory[3996].date,events:[]}}
]};
for(const first of [0,3500]){
 const result=patternGeometry({_bars:fullHistory.slice(first)},historical);
 const pressure=result.boundaries.find(x=>x.name==='已验证压力区间').points;
 const extension=result.boundaries.find(x=>x.name.startsWith('颈线延伸')).points;
 assert.equal(pressure[0].index,3710-first);
 assert(Math.abs(pressure[0].price-3.20)<1e-9,'the line must start at the stored 3.20 upper pivot');
 assert(Math.abs(pressure[1].price-(3.12+4*triangleLine.slope))<1e-9);
 assert.deepEqual(pressure[1],extension[0],'the frozen extension shares the same coordinate origin');
}
