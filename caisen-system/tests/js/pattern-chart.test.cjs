const assert = require('node:assert/strict');
const {patternGeometry,clipPoints,mainOverlays,patternDetails,instanceLifecycle} = require('../../caisen/static/pattern-chart.js');
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
const pendingTerminals={...shape,state:'forming',evidence:[
 {code:'target_reached',value:null},{code:'structure_invalidation',value:null},
 {code:'neckline_invalidation',value:null},{code:'neckline_projection_invalidation',value:null},
 {code:'followthrough_confirmed',value:null}
]};
assert.doesNotThrow(()=>patternGeometry({_bars:bars},pendingTerminals));
assert.throws(()=>patternGeometry({_bars:bars},{...pendingTerminals,evidence:[{code:'target_reached',value:{date:'1900-01-01',price:12}}]}),/缺少该形态结束位置的K线/);
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
assert(sg.boundaries.some(item=>item.name==='有效颈线至冻结'&&item.points[0].index===125&&item.points[1].index===131));
assert(sg.boundaries.some(item=>item.name==='冻结颈线延伸'&&item.points[0].index===131&&item.points[1].index===150));
assert(sg.markers.some(item=>item.name===`颈线有效起点 ${bars[125].date}`));
assert(sg.markers.some(item=>item.name===`颈线冻结日 ${bars[131].date}`));
assert(sg.markers.some(item=>item.name===`确认日 ${bars[150].date}`));
assert(sg.boundaries.some(item=>item.name==='备选压力线 2'));
const visibleMain=mainOverlays(sg);
assert.deepEqual(visibleMain.boundaries.map(item=>item.name),['有效颈线至冻结','冻结颈线延伸']);
assert(visibleMain.markers.some(item=>item.name==='确认日 '+bars[150].date));
assert(!visibleMain.markers.some(item=>item.name.includes('颈线测试')));
assert(!visibleMain.boundaries.some(item=>item.name.startsWith('备选压力线')));
console.log('main-bottom chart separates validated pressure range, frozen extension, and alternatives');
assert.deepEqual(clipPoints([{index:0,price:10},{index:100,price:20}],40,60),[{index:40,price:14},{index:60,price:16}]);
assert.deepEqual(clipPoints([{index:0,price:10},{index:20,price:12}],40,60),[]);
const facts=patternDetails({_bars:bars,_chart_offset:20},{pattern_type:'w_bottom',chart_offset:20,key_points:{pivot_1_index:4,pivot_1:8,pivot_1_date:bars[24].date,pivot_2_index:20,pivot_2:{price:12},pivot_3_index:34,pivot_3:9},evidence:[{code:'neckline_selection',value:{effective_start_date:bars[30].date,line_frozen_at:bars[50].date}}]});
assert.equal(facts.boundaryStart,bars[30].date);assert.equal(facts.freezeDate,bars[50].date);
assert.deepEqual(facts.points.map(point=>[point.name,point.date,point.price]),[['左底',bars[24].date,8],['颈线拐点',bars[40].date,12],['右底',bars[54].date,9]]);
const twoPointW=patternDetails({_bars:bars},{is_main_structure:true,pattern_type:'w_bottom',key_points:{pivot_1_index:24,pivot_1_date:bars[24].date,pivot_1:8,pivot_2_index:54,pivot_2_date:bars[54].date,pivot_2:9}});
assert.deepEqual(twoPointW.points.map(point=>point.name),['左底','右底']);
const twoPointWGeometry=patternGeometry({_bars:bars},{is_main_structure:true,pattern_type:'w_bottom',key_points:{pivot_1_index:24,pivot_1_date:bars[24].date,pivot_1:8,pivot_2_index:54,pivot_2_date:bars[54].date,pivot_2:9}});
assert.deepEqual(twoPointWGeometry.anchors.map(point=>point.name),['左底','右底']);
const threePointIhs=patternDetails({_bars:bars},{is_main_structure:true,pattern_type:'inverse_head_shoulders',key_points:{pivot_1_index:24,pivot_1_date:bars[24].date,pivot_1:8,pivot_2_index:40,pivot_2_date:bars[40].date,pivot_2:6,pivot_3_index:54,pivot_3_date:bars[54].date,pivot_3:8}});
assert.deepEqual(threePointIhs.points.map(point=>point.name),['左肩','头部','右肩']);
const mainW={...main,chart_offset:100,key_points:{pivot_1_index:2,pivot_1_date:bars[102].date,pivot_1:8,
 pivot_2_index:38,pivot_2_date:bars[138].date,pivot_2:9},evidence:[{code:'main_structure',value:{role:'main',support_lower:8,support_upper:9}}]};
assert.deepEqual(patternGeometry({_bars:bars,_chart_offset:20},mainW).anchors.map(p=>p.name),['左底','右底']);
const lifecyclePattern={...main,pattern_id:'instance-1',pattern_instance_id:'instance-1',
  instance_state_events:[
    {instance_id:'instance-1',event_id:'evt-1',event_seq:1,state:'forming',event_date:bars[125].date,entry_eligible:false},
    {instance_id:'instance-1',event_id:'evt-2',event_seq:2,state:'confirmed',event_date:bars[150].date,entry_eligible:true},
    {instance_id:'instance-1',event_id:'evt-3',event_seq:3,state:'focus',event_date:bars[160].date,entry_eligible:false},
    {instance_id:'instance-1',event_id:'evt-4',event_seq:4,state:'invalidated',event_date:bars[170].date,entry_eligible:false}
  ],evidence:[...main.evidence,{code:'neckline_selection',value:{selected_candidate_id:'boundary-7',
    line_frozen_at:bars[130].date,selected_line:{slope:0,intercept:12}}}]};
const lifecycle=instanceLifecycle(lifecyclePattern);
assert.deepEqual(lifecycle.map(item=>[item.kind,item.date]),[
  ['discovered',bars[125].date],['boundary_frozen',bars[130].date],['confirmed',bars[150].date],
  ['focus',bars[160].date],['invalidated',bars[170].date]]);
const weakLifecycle=instanceLifecycle({...lifecyclePattern,instance_state_events:[
  {instance_id:'instance-1',event_id:'evt-forming',event_seq:1,state:'forming',event_date:bars[125].date,entry_eligible:false},
  {instance_id:'instance-1',event_id:'evt-weak',event_seq:2,state:'forming',event_type:'weak_crossing',
    event_date:bars[140].date,entry_eligible:false}
]});
assert.ok(weakLifecycle.some(item=>item.kind==='weak_crossing'&&item.date===bars[140].date));

const targetLifecycle=instanceLifecycle({...lifecyclePattern,
  instance_state_events:[...lifecyclePattern.instance_state_events.slice(0,2),
    {instance_id:'instance-1',event_id:'evt-target',event_seq:3,state:'target_reached',
      event_date:bars[165].date,entry_eligible:false}]});
assert.ok(targetLifecycle.some(item=>item.kind==='target_reached'&&item.date===bars[165].date));
const lifecycleGeometry=patternGeometry({_bars:bars},lifecyclePattern);
assert(lifecycleGeometry.markers.some(item=>item.name==='结构与边界配对冻结'&&item.instance_id==='instance-1'));
assert(mainOverlays(lifecycleGeometry).markers.some(item=>item.name==='首次突破确认 · BUY'));
