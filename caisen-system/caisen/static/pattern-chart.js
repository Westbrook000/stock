(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.CaisenPatternChart=api})(typeof globalThis!=='undefined'?globalThis:this,function(){
  const labels={bull_flag:['推动起点','前高','整理低点1','整理高点','整理低点2'],bear_flag:['推动起点','前低','整理高点1','整理低点','整理高点2'],w_bottom:['左底','颈线拐点','右底'],m_top:['左顶','颈线拐点','右顶'],inverse_head_shoulders:['左肩','颈线左点','头部','颈线右点','右肩'],head_shoulders:['左肩','颈线左点','头部','颈线右点','右肩']};
  const mainLabels={w_bottom:['左底','右底'],inverse_head_shoulders:['左肩','头部','右肩'],triangle_bottom:['左底','上折点1','中底','上折点2','右底']};
  function patternGeometry(data,pattern){
    const bars=data._bars||[],offset=Number.isInteger(pattern.chart_offset)?pattern.chart_offset:(data._chart_offset||0),points=pattern.key_points||{};
    const main=Boolean((pattern.evidence||[]).some(item=>item.code==='main_structure'&&item.value?.role==='main'));
    const pointLabels=main?mainLabels[pattern.pattern_type]:labels[pattern.pattern_type];
    const byDate=new Map(bars.map((bar,i)=>[bar.date,i]));
    const indexOf=(value)=>value?.date?byDate.get(value.date):(Number.isInteger(value?.index)?value.index+offset:undefined);
    const anchors=Object.entries(points).filter(([key,value])=>/^pivot_\d+_index$/.test(key)&&Number.isInteger(value)).map(([key,value])=>{
      const ordinal=Number(key.match(/\d+/)[0])-1;
      const date=points[key.replace('_index','_date')]||points[key.replace('_index','')]?.date;
      return {index:date?byDate.get(date):value+offset,price:Number(points[key.replace('_index','')]?.price??points[key.replace('_index','')]??points[key.replace('_index','_price')]),name:(pointLabels?.length===Object.keys(points).filter(key=>/^pivot_\d+_index$/.test(key)).length?pointLabels[ordinal]:null)||`结构点${ordinal+1}`};
    }).sort((a,b)=>a.index-b.index);
    if(!anchors.length||anchors.some(p=>!bars[p.index]||!Number.isFinite(p.price)))throw new Error('缺少该历史形态对应的K线或结构点');
    const markers=[],evidence=pattern.evidence||[];
    const confirmation=evidence.find(e=>e.code==='close_confirmation')?.value;
    const ci=pattern.confirmation_date?byDate.get(pattern.confirmation_date):undefined;
    if(ci!==undefined&&confirmation!==false)markers.push({index:ci,price:bars[ci].close,name:'价格确认'});
    const terminalNames={target_reached:'目标已达',structure_invalidation:'结构失效',neckline_invalidation:'颈线失效',neckline_projection_invalidation:'颈线失效',followthrough_confirmed:'后续边界确认'};
    for(const e of evidence){if(!terminalNames[e.code])continue;const index=indexOf(e.value);if(index===undefined||!bars[index])throw new Error('缺少该形态结束位置的K线');markers.push({index,price:Number(e.value.price??e.value.level??bars[index].close),name:terminalNames[e.code]})}
    for(const e of evidence){
      const value=e.value||{};
      const tests=e.code==='neckline_test'?[value]:e.code==='neckline_tests'?(Array.isArray(value)?value:[]):[];
      for(const test of tests){const index=byDate.get(test.pivot_date||test.date);if(index!==undefined)markers.push({index,price:Number(test.level??bars[index].close),name:`颈线测试 · ${test.group_id||'未分组'}`})}
      if(e.code==='neckline_breakout'){const index=byDate.get(value.date);if(index!==undefined&&!markers.some(marker=>marker.index===index&&marker.name.includes('确认')))markers.push({index,price:Number(value.close??bars[index].close),name:'颈线收盘确认'})}
      if(e.code==='weak_crossings'){for(const crossing of Array.isArray(value)?value:[]){const index=byDate.get(crossing.date);if(index!==undefined)markers.push({index,price:Number(crossing.close??crossing.price??bars[index].close),name:'弱穿越'})}}
    }
    if(pattern.state==='invalidated'&&!markers.some(m=>m.name.includes('失效'))){
      const value=evidence.find(e=>e.code==='data_discontinuity')?.value,index=indexOf(value);
      if(index!==undefined&&bars[index])markers.push({index,price:Number(value.level??bars[index].close),name:'行情分段结束'});
    }
    const start=anchors[0].index,end=Math.max(...anchors.map(p=>p.index),...markers.map(p=>p.index));
    const pad=Math.max(10,Math.ceil((end-start+1)*.12));
    const neckline=pattern.neckline?Array.from({length:end-start+1},(_,i)=>({index:start+i,price:pattern.neckline.slope*(start+i-offset)+pattern.neckline.intercept})):[];
    const boundaries=neckline.length?[{name:'颈线 / 边界',points:neckline}]:[];
    const values=Object.fromEntries(evidence.map(e=>[e.code,e.value]));
    if(Number.isFinite(values.box_upper)&&Number.isFinite(values.box_lower)){
      boundaries.length=0;
      for(const [key,name] of [['box_upper','箱体上沿'],['box_lower','箱体下沿']])boundaries.push({name,points:[{index:start,price:values[key]},{index:end,price:values[key]}]});
    }else if(pattern.pattern_type.startsWith('triangle_')||(pattern.pattern_type.endsWith('_flag')&&anchors.length===5)){
      const a=pattern.pattern_type.endsWith('_flag')?anchors[2]:anchors[0],b=anchors.at(-1);
      if(b.index>a.index){const slope=(b.price-a.price)/(b.index-a.index);boundaries.push({name:'另一侧结构边界',points:Array.from({length:end-start+1},(_,i)=>({index:start+i,price:a.price+slope*(start+i-a.index)}))})}
    }
    const mainStructure=values.main_structure||{};
    if(mainStructure.support_lower!==undefined&&mainStructure.support_upper!==undefined){
      for(const [price,name] of [[mainStructure.support_lower,'主结构支撑下沿'],[mainStructure.support_upper,'主结构支撑上沿']])boundaries.push({name,points:[{index:start,price:Number(price)},{index:end,price:Number(price)}]});
    }
    if(Number.isFinite(Number(mainStructure.neckline))&&!pattern.neckline)boundaries.push({name:'冻结颈线',points:[{index:start,price:Number(mainStructure.neckline)},{index:end,price:Number(mainStructure.neckline)}]});
    return {anchors,markers,neckline,boundaries,start,end,first:Math.max(0,start-pad),last:Math.min(bars.length-1,end+pad)};
  }
  return {patternGeometry};
});
